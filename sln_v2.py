import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
import pdb

import sys
sys.path.append('.')
from print_colored_text import print_colored_text


class SLN:
    def __init__(self, 
                 right_model_checkpoint_name, 
                 left_model_checkpoint_name, 
                 verbose=True, 
                 return_type1_answer=False, 
                 return_highest_valence=True, 
                 return_all_IDLs=False,
                 round_valence=True,
                 decrement_future_negative_logits_with_rewards=False,
                 add_thinking_space=False,
                 left_model_device="cuda:0",
                 right_model_device="cuda:0",
                 trajectories_per_IDL=1,
                 temperature=0.7
                ):
      with torch.no_grad():  
        self.right_model_checkpoint_name = right_model_checkpoint_name
        self.left_model_checkpoint_name = left_model_checkpoint_name
        self.return_type1_answer = return_type1_answer
        self.return_highest_valence = return_highest_valence # if false, returns LAST IDL as the answer vs. highest valence
        self.return_all_IDLs = return_all_IDLs
        self.round_valence = round_valence
        self.decrement_future_negative_logits_with_rewards = decrement_future_negative_logits_with_rewards
        self.add_thinking_space = add_thinking_space
        self.verbose = verbose
        
        self.left_model_device = left_model_device
        self.right_model_device = right_model_device
        
        self.model_tok = '<left model>'
        self.pad_tok = '[PAD]'
        self.max_ctx_len = 2000
        self.valence_input_baseline = 0.5
        self.valence_input_alpha = 2.
        self.base_model_id = "EleutherAI/pythia-410m"
        self.trajectories_per_IDL = trajectories_per_IDL
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.pad_tok})
        print('Setting tokenizer.pad_token_id to: ' + str(self.tokenizer.pad_token_id) + " with token: "+self.pad_tok)

        # Initialize right model
        if not self.right_model_checkpoint_name:
            raise ValueError("Right model checkpoint name must be provided")
        
        print("LOADING RIGHT MODEL FROM: " + self.right_model_checkpoint_name)
        checkpoint = torch.load(self.right_model_checkpoint_name, map_location=torch.device(self.right_model_device))
        self.current_right_model = AutoModelForCausalLM.from_pretrained(self.base_model_id).to(self.right_model_device)
        self.current_right_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.current_right_model.resize_token_embeddings(len(self.tokenizer))
        self.current_right_model.load_state_dict(checkpoint['model_state_dict'])

        vocab_size = self.current_right_model.config.vocab_size
        self.valence_layer = nn.Sequential(
            nn.LayerNorm(vocab_size, self.current_right_model.config.layer_norm_eps),
            nn.Linear(vocab_size, 2),
            nn.Softmax(dim=-1)
        ).to(self.right_model_device)
        try:
            self.valence_layer.load_state_dict(checkpoint['valence_layer_state_dict'])
        except Exception as e:
            try:
                self.valence_layer.load_state_dict(checkpoint['reward_layer_state_dict'])
            except Exception as e:
                print("could not load valence_layer_state_dict or reward_layer_state_dict in right model: " + str(e))
                exit()
        # Initialize left model
        if not self.left_model_checkpoint_name:
            raise ValueError("Left model checkpoint name must be provided")
        print("LOADING LEFT MODEL FROM: " + self.left_model_checkpoint_name)
        
        checkpoint = torch.load(self.left_model_checkpoint_name, map_location=torch.device(self.left_model_device))
        self.current_left_model = AutoModelForCausalLM.from_pretrained(self.base_model_id).to(self.left_model_device)
        self.current_left_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.current_left_model.resize_token_embeddings(len(self.tokenizer))
        self.current_left_model.load_state_dict(checkpoint['model_state_dict'])
        print("consciousness booted! Give me a prompt:")

    def _forward_right(self, input_ids, attention_mask):
        with torch.no_grad():  
            input_ids = input_ids.to(self.right_model_device)
            attention_mask = attention_mask.to(self.right_model_device)
            logits = self.current_right_model(input_ids, attention_mask=attention_mask, return_dict=True).logits
            outputs = self.valence_layer(logits) # which is batch x seq x 2 (the second channel is the positive valence )
            
            if self.round_valence:
                valence_mask = torch.round(softmax(outputs, dim=-1))[0,:,1]
            else:
                valence_mask = softmax(outputs, dim=-1)[0,:,1]
            return valence_mask

    def _forward_left(self, input_ids, input_valence, attention_mask, zero_out_bit_flag=None):
        with torch.no_grad():  
            input_ids = input_ids.to(self.left_model_device)
            attention_mask = attention_mask.to(self.left_model_device)
            input_valence = input_valence.to(self.left_model_device)
            
            # valence_masks_tensors_padded = nn.utils.rnn.pad_sequence([input_valence], batch_first=True, padding_value=0)
    
            # Ensure the padded sequences have the desired length
            # valence_masks_tensors_padded = valence_masks_tensors_padded[:, :self.max_ctx_len]
            # if valence_masks_tensors_padded.size(1) < self.max_ctx_len:
            #     padding = torch.zeros((valence_masks_tensors_padded.size(0), self.max_ctx_len - valence_masks_tensors_padded.size(1))).to(self.left_model_device)
            #     valence_masks_tensors_padded = torch.cat([valence_masks_tensors_padded, padding], dim=1)
            
            model_outputs = self._forward_left_with_valence_input(
                self.current_left_model,
                input_ids,
                #valence_masks_tensors_padded,
                input_valence,
                attention_mask=attention_mask,
                return_dict=True,
                alpha=self.valence_input_alpha,
                baseline=self.valence_input_baseline
            )
    
            model_outputs = model_outputs['logits']  

            if self.decrement_future_negative_logits_with_rewards:
                # model_outputs (seq x Vocab), we want to -INF all logits that we know have 0 valence
                # TODO: Do we want to do float('-inf') or do we want to do something less severe like -1000... hmm..
                model_outputs = model_outputs.masked_fill(zero_out_bit_flag.bool().to(model_outputs.device), float('-inf'))

            generated_tokens = []
            for i in range(self.trajectories_per_IDL):
                # generated_tokens = torch.argmax(model_outputs, dim=-1) # TODO: greedy sampling?? or be smarter..
                    
                # Apply softmax to convert logits to probabilities
                probabilities = torch.softmax(model_outputs / self.temperature, dim=-1)
                
                # Sample from the probabilities
                sampled_tokens = torch.multinomial(probabilities.view(-1, probabilities.size(-1)), 1)
                
                # Reshape the sampled tokens to match the original shape
                sampled_tokens = sampled_tokens.view(probabilities.size()[:-1])
                
                generated_tokens.append(sampled_tokens)
            
            # Concatenate all the generated tokens along the first dimension
            return torch.cat(generated_tokens, dim=0)

    def _forward_left_with_valence_input(self, 
                                        current_left_model, 
                                        input_ids, 
                                        token_valences, 
                                        attention_mask=None, 
                                        return_dict=True, 
                                        alpha=1., 
                                        baseline=0.5):
        with torch.no_grad():  
            original_embeddings = current_left_model.get_input_embeddings()
            embeddings = original_embeddings(input_ids)
    
            token_valences = (token_valences.clone().float() - baseline)*alpha
            # token_valences = token_valences.expand(-1, -1, embeddings.size(-1))
            # modified_embeddings = embeddings + token_valences
            embeddings[:, :, -1] = token_valences
        
            logits = current_left_model(
              inputs_embeds=embeddings,
              attention_mask=attention_mask,
              return_dict=return_dict
            ).logits
    
            if not return_dict:
                return logits
    
            return {"logits": logits}

    def _stopping_criteria(self):
        # return True if we should stop IDL, False if we should not
        if self.IDL_count > self.IDL_limit:
            
            return False #break out of IDL
        else:
            return True #continue with IDL

        if sum(self.last_valence_mask) >= 0.98 * len(self.last_valence_mask): # meaning mostly 1s valence in the left response. 
            return False #we got a good answer! Break out of IDL! 
        return True #continue with IDL.. not there yet.. 
        
    def _decode(self, output_tokens):
        tokens = output_tokens.cpu().numpy()
        txt = []
        for i in range(tokens.shape[0]):
            txt.append(self.tokenizer.decode(output_tokens.cpu().numpy()[i], skip_special_tokens=False))
        return txt 


    def forward(self, prompt_text):
        with torch.no_grad():  
            # TODO what does padding and truncation do? remove for now..
            self.prompt_tokens = self.tokenizer(self.tokenizer.eos_token + " " + prompt_text + " " + self.model_tok + " ", 
                                    return_tensors="pt", 
                                    padding=False, 
                                    truncation=False, 
                                    add_special_tokens=True)
            
            # Check if bos_token_id is not the first token and prepend it if necessary
            # if self.prompt_tokens.input_ids[0, 0] != self.tokenizer.bos_token_id:
            #     print("adding bos tok to input_ids")
            #     bos_token_tensor = torch.tensor([[self.tokenizer.bos_token_id]])
            #     self.prompt_tokens.input_ids = torch.cat([bos_token_tensor, self.prompt_tokens.input_ids], dim=1)
            #     self.prompt_tokens.attention_mask = torch.cat([torch.tensor([[1]]), self.prompt_tokens.attention_mask], dim=1)
            # if self.prompt_tokens.input_ids[0, -1] != self.tokenizer.eos_token_id:
            #     eos_token_tensor = torch.tensor([[self.tokenizer.eos_token_id]])
            #     self.prompt_tokens.input_ids = torch.cat([self.prompt_tokens.input_ids,eos_token_tensor], dim=1)
            IDLs = []
            if self.verbose:
                print('generating Type 1 response:')
    
            self.output_tokens_full = self.current_left_model.generate(
                self.prompt_tokens.input_ids.to(self.left_model_device),
                attention_mask=self.prompt_tokens.attention_mask.to(self.left_model_device),
                max_new_tokens = 100, #self.max_ctx_len - len(prompt_text),
                #max_length=self.max_ctx_len/2,  # Maximum length of the sequence
                eos_token_id=-1,#self.tokenizer.eos_token_id,  # Stop when the end-of-sequence token is generated
                num_beams=5,  # Use beam search with 5 beams
                # early_stopping=True,  # Stop early when the beams are sufficiently similar
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,  # Enable sampling
                temperature=0.7,  # Control the randomness of sampling
                num_return_sequences=self.trajectories_per_IDL  # Number of samples to generate
            )
            
            num_generated_tokens = self.output_tokens_full.shape[1] - self.prompt_tokens.input_ids.shape[1]
            self.output_tokens = self.output_tokens_full[:,-num_generated_tokens:]
            

            # TODO: DO WE WANT TO ADD MORE "SPACE" FOR THE MODEL TO THINK? ADD PADDING! 
            if num_generated_tokens < 200 and self.add_thinking_space:
                size_of_padding = 200 - num_generated_tokens

                # Create a tensor of pad_token_id with the required padding size
                padding = torch.full((1,size_of_padding), 
                                     self.tokenizer.pad_token_id, 
                                     dtype=self.output_tokens.dtype
                                    ).to(self.output_tokens.device)
                
                # Concatenate the original output_tokens with the padding
                self.output_tokens_full = torch.cat((self.output_tokens_full, padding), dim=-1)
                self.output_tokens = torch.cat((self.output_tokens, padding), dim=-1)
                
                # Update num_generated_tokens
                num_generated_tokens += size_of_padding
                
            if self.decrement_future_negative_logits_with_rewards:
            
                # zero_out_bit_flag[i,j] = 1 iff we want to zero out the jth vocab id at the ith token in the seq, 0 otherwise. 
                # Start with none. Then add 1s to zero_out_bit_flag as we get valence.
                zero_out_bit_flag = torch.zeros( (self.output_tokens_full.shape[1], len(self.tokenizer) ))
                
            
            else:
                zero_out_bit_flag = None
            
            
            self.last_left_model_response = self._decode(self.output_tokens) 
            self.last_left_model_response_full = self._decode(self.output_tokens_full) 
            if self.return_type1_answer:
                return self.last_left_model_response
            
            self.last_valence_mask = []
            self.IDL_count = 0
            self.IDL_limit = 10  # This should be set according to your needs
            
            highest_valence = -1.

            while True:
                
                # inputs = self.tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_ctx_len, add_special_tokens=True)
                # input_ids = inputs['input_ids']
                sample_tracker = []
               
                for i in range(self.trajectories_per_IDL):
                    input_ids = torch.cat([self.prompt_tokens.input_ids.to(self.right_model_device), self.output_tokens[i,:].unsqueeze(0).to(self.right_model_device)], dim=-1)
                    
                    attention_mask = torch.ones_like(input_ids).to(self.right_model_device)
        
                    # Forward pass right model
                    self.last_valence_mask = self._forward_right(input_ids, attention_mask)
    
                    if self.decrement_future_negative_logits_with_rewards:
                        # Update zero_out_bit_flag with 1s where there is 0 valence for specific toks in specific seq positions
                        
                        indices_to_update = torch.where(self.last_valence_mask == 0)[0]
                        zero_out_bit_flag[indices_to_update, input_ids[0,indices_to_update]] = 1
                    
                    current_valence = int(100.0*sum(self.last_valence_mask) / len(self.last_valence_mask))
                    sample_tracker.append((current_valence,input_ids,self.last_valence_mask))
                if self.verbose:
                    for current_valence, input_ids, last_valence_mask in sample_tracker:
                        print(f"IDL count {self.IDL_count}: current_valence: {current_valence } on : {len(last_valence_mask) } IDL: ", end='')
                        print_colored_text(last_valence_mask, input_ids, self.tokenizer)
                        print("------------------------------------")
                
                current_valence, input_ids, last_valence_mask = max(sample_tracker, key = lambda x: x[0])
                
                best_left_model_response_for_this_IDL = self._decode(input_ids)[0]
                
                if self.return_all_IDLs:
                    # we only return the best sample per IDL
                    IDLs.append( ( {
                                    "IDL_count": self.IDL_count,
                                    "left_model_response": best_left_model_response_for_this_IDL,
                                    "right_model_valence": last_valence_mask
                                   }
                                 )
                               )
                
                if current_valence > highest_valence:
                    highest_valence = current_valence
                    self.highest_valence_response = best_left_model_response_for_this_IDL
                    best_input_ids = input_ids

                
                
                if not self._stopping_criteria():
                    # score is high enough or we have hit our limit
                    break
                
                # score not high enough.. we need to keep trying
                if self.verbose:
                    print('Now doing Type 2 thinking to really think about it...')
            
                # Forward pass left model and update our output_tokens with the new valence scores
                input_ids = input_ids.to(self.left_model_device)
                self.last_valence_mask = last_valence_mask.to(self.left_model_device)
                attention_mask = attention_mask.to(self.left_model_device)

                
                self.output_tokens_full = self._forward_left(input_ids, self.last_valence_mask, attention_mask, zero_out_bit_flag=zero_out_bit_flag)
                
                    
                self.output_tokens = self.output_tokens_full[:,-num_generated_tokens:]
                
                # also, we may find the first padding tok, and eliminate everything after it? 
                # first_pad_tok_index = torch.nonzero(torch.eq(self.output_tokens, 
                #                                self.tokenizer.pad_token_id), 
                #                                as_tuple=True)[1][0].item()
                
                
                #TODO: CHECK IF WE HAVE TO REMOVE [0] or not.
                
                # self.last_left_model_response = self.tokenizer.decode(self.output_tokens[0], skip_special_tokens=True)
                
                self.last_left_model_response = self._decode(self.output_tokens) 
                self.last_left_model_response_full = self._decode(self.output_tokens_full) 
            
                #             response_parts = self.last_left_model_response.split(self.model_tok)
                #             if len(response_parts) > 1:
                #                 self.last_left_model_response = response_parts[-1]
                #             else:
                #                 raise ValueError(f"The response does not contain the model token: {self.model_tok} and model response: {self.last_left_model_response}")
    
                
                self.IDL_count += 1

            if self.return_highest_valence:
                return self.highest_valence_response
            if self.return_all_IDLs:
                return IDLs
            return best_left_model_response_for_this_IDL


if __name__ == "__main__":
    print("Booting consciousness... one sec.. :)")
    left_model_checkpoint = "/left_checkpoints/left_checkpoint_20240715113638_iter_700_loss_43.06.pth" #"<path to right model checkpoint>"
    # right_model_checkpoint = "/right_checkpoints/right_checkpoint_20240715155144_iter_10_loss_6.31.pth" #"<path to left model checkpoint>"
    # left_model_checkpoint = "./left_checkpoint_20240711093303_iter_100_loss_55.50.pth" #"<path to right model checkpoint>"
    right_model_checkpoint = "/root/right_checkpoint_20240709113005_iter_2000_loss_0.52.pth" #"<path to left model checkpoint>"
    
    sln = SLN(right_model_checkpoint, left_model_checkpoint)
  
    prompt_text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    
    target_response = "Let's break this down step by step.\n\n**Step 1: Understand the problem**\nNatalia sold clips to 48 friends in April. Then, she sold half as many clips in May. We need to find the total number of clips she sold in April and May.\n\n**Step 2: Calculate the number of clips sold in May**\nIf Natalia sold half as many clips in May as she did in April, that means she sold:\n\n48 (clips sold in April) / 2 = 24 clips in May\n\n**Step 3: Add the number of clips sold in April and May**\nTo find the total number of clips sold, we add the number of clips sold in April and May:\n\n48 (clips sold in April) + 24 (clips sold in May) = 72\n\n**Conclusion**\nNatalia sold a total of 72 clips in April and May. #### 72"

    model_response = sln.forward(prompt_text)
    
    print("Target Response:", target_response)
    print("="*50)
    print("Model Response:", model_response)
    
