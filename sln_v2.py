import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
import pdb

class SLN:
    def __init__(self, right_model_checkpoint_name, left_model_checkpoint_name):
      with torch.no_grad():  
        self.right_model_checkpoint_name = right_model_checkpoint_name
        self.left_model_checkpoint_name = left_model_checkpoint_name
        
        self.left_model_device = "cuda:0"
        self.right_model_device = "cuda:0"
        
        self.model_tok = '<left model>'
        self.pad_tok = '[PAD]'
        self.max_ctx_len = 2000
        self.valence_input_baseline = 0.5
        self.valence_input_alpha = 2.
        self.base_model_id = "EleutherAI/pythia-410M"

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
            #valence_mask = torch.round(softmax(outputs, dim=-1))[0,:,1]
            valence_mask = softmax(outputs, dim=-1)[0,:,1]
            return valence_mask

    def _forward_left(self, input_ids, input_valence, attention_mask):
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
            generated_tokens = torch.argmax(model_outputs, dim=-1) # TODO: greedy sampling?? or be smarter..
            
            return generated_tokens

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
    
            token_valences = (token_valences.unsqueeze(-1).float() - baseline) * alpha
            token_valences = token_valences.expand(-1, embeddings.size(-1))
            modified_embeddings = embeddings + token_valences
    
            logits = current_left_model(
                inputs_embeds=modified_embeddings,
                attention_mask=attention_mask,
                return_dict=return_dict
            ).logits
    
            if not return_dict:
                return logits
    
            return {"logits": logits}

    def _stopping_criteria(self):
        # return True if we should stop IDL, False if we should not
        if self.IDL_count > self.IDL_limit:
            
            self.last_left_model_response = self.last_left_model_response + "|||| .... I do not know.. I am not smart enough yet!"
            return False #break out of IDL
        else:
            return True #continue with IDL

        if sum(self.last_valence_mask) >= 0.98 * len(self.last_valence_mask): # meaning mostly 1s valence in the left response. 
            return False #we got a good answer! Break out of IDL! 
        return True #continue with IDL.. not there yet.. 
        
    def _decode(self, output_tokens):
        txt = self.tokenizer.decode(output_tokens.cpu().numpy()[0], skip_special_tokens=True)
        return txt 
        

    def forward(self, prompt_text):
        with torch.no_grad():  
            # TODO what does padding and truncation do? remove for now..
            self.prompt_tokens = self.tokenizer(prompt_text + self.model_tok, return_tensors="pt") #padding=True, truncation=True)
            
            print('generating Type 1 response:')
    
            self.output_tokens = self.current_left_model.generate(
                self.prompt_tokens.input_ids.to(self.left_model_device),
                attention_mask=self.prompt_tokens.attention_mask.to(self.left_model_device),
                max_new_tokens = 100, #self.max_ctx_len - len(prompt_text),
                #max_length=self.max_ctx_len/2,  # Maximum length of the sequence
                eos_token_id=self.tokenizer.eos_token_id,  # Stop when the end-of-sequence token is generated
                num_beams=5,  # Use beam search with 5 beams
                early_stopping=True,  # Stop early when the beams are sufficiently similar
                pad_token_id=self.tokenizer.pad_token_id
            )
    
            num_generated_tokens = self.output_tokens.shape[1] - self.prompt_tokens.input_ids.shape[1]
            
            self.output_tokens = self.output_tokens[-num_generated_tokens:]
            self.last_left_model_response = self._decode(self.output_tokens) #self.tokenizer.decode(self.last_left_model_response.cpu().numpy()[0], skip_special_tokens=True)
            
            # self.last_left_model_response = self.last_left_model_response.split(self.model_tok)[-1]
            # self.last_valence_mask_subset = []
            
            self.last_valence_mask = []
            self.IDL_count = 0
            self.IDL_limit = 10  # This should be set according to your needs
            
            
            print('Now doing Type 2 thinking to really think about it...')
            
            # TODO: DO WE WANT TO ADD MORE "SPACE" FOR THE MODEL TO THINK? ADD PADDING! 
            # size_of_padding = 40
            # num_generated_tokens = num_generated_tokens + size_of_padding
            # self.output_tokens = self.output_tokens.extend(self.tokenizer.pad_token_id * size_of_padding)
            
            while True:
                
                # inputs = self.tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_ctx_len, add_special_tokens=True)
                # input_ids = inputs['input_ids']
                
                input_ids = torch.cat([self.prompt_tokens.input_ids.to(self.right_model_device), self.output_tokens.to(self.right_model_device)], dim=-1)
                
                attention_mask = torch.ones_like(input_ids).to(self.right_model_device)
    
                # Forward pass right model
                self.last_valence_mask = self._forward_right(input_ids, attention_mask)
                one_line = ' '.join(map(str, self.last_valence_mask.tolist()))

                print(f"IDL count {self.IDL_count}: total_valence: {sum(self.last_valence_mask) / len(self.last_valence_mask) } IDL: {self.last_left_model_response} per_tok_valence: {one_line}")
    
                
                if not self._stopping_criteria():
                    # score is high enough or we have hit our limit
                    break
                
                # score not high enough.. we need to keep trying
                
                # Forward pass left model and update our output_tokens with the new valence scores
                input_ids = input_ids.to(self.left_model_device)
                self.last_valence_mask = self.last_valence_mask.to(self.left_model_device)
                attention_mask = attention_mask.to(self.left_model_device)
                
                self.output_tokens = self._forward_left(input_ids, self.last_valence_mask, attention_mask)
                self.output_tokens = self.output_tokens[-num_generated_tokens:]
                # also, we may find the first padding tok, and eliminate everything after it? 
                # first_pad_tok_index = torch.nonzero(torch.eq(self.output_tokens, 
                #                                self.tokenizer.pad_token_id), 
                #                                as_tuple=True)[1][0].item()


                
                #TODO: CHECK IF WE HAVE TO REMOVE [0] or not.
                self.last_left_model_response = self.tokenizer.decode(self.output_tokens[0], skip_special_tokens=True)
    
                #             response_parts = self.last_left_model_response.split(self.model_tok)
                #             if len(response_parts) > 1:
                #                 self.last_left_model_response = response_parts[-1]
                #             else:
                #                 raise ValueError(f"The response does not contain the model token: {self.model_tok} and model response: {self.last_left_model_response}")
    
                
                self.IDL_count += 1
            return self.last_left_model_response


if __name__ == "__main__":
    print("Booting consciousness... one sec.. :)")
    left_model_checkpoint = "/left_checkpoints/left_checkpoint_20240715173212_iter_800_loss_37.63.pth" #"<path to right model checkpoint>"
    # right_model_checkpoint = "/right_checkpoints/right_checkpoint_20240715155144_iter_10_loss_6.31.pth" #"<path to left model checkpoint>"
    # left_model_checkpoint = "./left_checkpoint_20240711093303_iter_100_loss_55.50.pth" #"<path to right model checkpoint>"
    right_model_checkpoint = "right_checkpoint_20240709113005_iter_2000_loss_0.52.pth" #"<path to left model checkpoint>"
    
    sln = SLN(right_model_checkpoint, left_model_checkpoint)
  
    prompt_text = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    target_response = "Let's break this down step by step.\n\n**Step 1: Understand the problem**\nNatalia sold clips to 48 friends in April. Then, she sold half as many clips in May. We need to find the total number of clips she sold in April and May.\n\n**Step 2: Calculate the number of clips sold in May**\nIf Natalia sold half as many clips in May as she did in April, that means she sold:\n\n48 (clips sold in April) / 2 = 24 clips in May\n\n**Step 3: Add the number of clips sold in April and May**\nTo find the total number of clips sold, we add the number of clips sold in April and May:\n\n48 (clips sold in April) + 24 (clips sold in May) = 72\n\n**Conclusion**\nNatalia sold a total of 72 clips in April and May. "

    model_response = sln.forward(prompt_text)
    
    print("Target Response:", target_response)
    print("="*50)
    print("Model Response:", model_response)
    
