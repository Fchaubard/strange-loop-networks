import torch
from transformers import GPTNeoXModel, GPTNeoXForCausalLM, GPTNeoXConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from copy import deepcopy
from typing import Optional, Tuple, Union
from torch import nn, optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
from typing import Optional, Tuple, Union
import random
import numpy as np
import pdb


class StrangeLoopNetwork(nn.Module):
    
    def __init__(self, 
                 base_model: AutoModelForCausalLM, 
                 tokenizer: AutoTokenizer, 
                 IDL_stopping_probability_thresh = 0.3,
                 IDL_limit = 3,
                 max_internal_seq_length = 500,
                 baseline_score = 0.5,
                 max_grad_norm = 0.5,
                 epsilon = 0.2,
                 max_microbatch_size = 20,
                 lr = 5e-5,
                 training_mode=True):
        
        super().__init__()
        
        self.left_model = base_model
        self.right_model = deepcopy(base_model)
        self.tokenizer = tokenizer

        # Define the valence layer
        vocab_size = base_model.config.vocab_size
        
        self.left_model_device=0
        self.right_model_device=1
        
        self.left_model.to(f'cuda:{self.left_model_device}')
        self.right_model.to(f'cuda:{self.right_model_device}')
        
        self.valence_layer = nn.Sequential(
            nn.LayerNorm(vocab_size, base_model.config.layer_norm_eps),
            nn.Linear(vocab_size, 2),
            nn.Softmax(dim=-1)
        ).to(f'cuda:{self.right_model_device}')
        
        self.IDL_trajectory_pairs = []
        
        self.IDL_limit = IDL_limit
        self.IDL_stopping_probability_thresh = IDL_stopping_probability_thresh
        self.max_internal_seq_length = max_internal_seq_length
        
        self.baseline_score = baseline_score # to offset valence [0-1] for classificaiton (i.e. 0.5 so valence is -0.5 to 0.5)
        self.max_grad_norm = max_grad_norm # for clip loss
        self.epsilon = epsilon # for clip loss
        self.max_microbatch_size = max_microbatch_size # to make sure we do not run out of memory
        
        if training_mode:
            self.optimizer_left = optim.AdamW(self.left_model.parameters(), lr=lr)
            self.optimizer_right = optim.AdamW(list(self.right_model.parameters()) + list(self.valence_layer.parameters()), lr=lr) 
            self.left_model.train()
            self.right_model.train()
            

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cache_trajectory_pairs: bool = True,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        self.IDL_trajectory_pairs = []
        print("Fpass SLN")
        
        if inputs_embeds is None and input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")
        
        x = deepcopy(input_ids)
        iterr_ = 0
        for _ in range(self.IDL_limit): 
            
            x = self.add_special_tok_ids(x, bos_only=True)
            
            x_left = self.left_model.generate(x.to(f'cuda:{self.left_model_device}'), max_length=self.max_internal_seq_length)
            
            x_left = self.cut_all_after_eos_token(x_left[0]).unsqueeze(0) # TODO,add temperature
            
            x_left_only = x_left[:,input_ids.shape[-1]:] 
            
            x_right = torch.cat([input_ids.to(f'cuda:{self.left_model_device}'), x_left_only.to(f'cuda:{self.left_model_device}')], dim=-1)
            
            x_right = self.add_special_tok_ids(x_right)
            
            outputs_right = self.right_model(x_right.to(f'cuda:{self.right_model_device}'), return_dict=True)
            
            y_hat_right = self.valence_layer(outputs_right.logits).to(f'cpu:{0}')

            x_score = y_hat_right[:, :, 1].mean() - self.baseline_score
            
            scalar_value = x_score.item()
            
            string_value = f"<valence:{scalar_value:.2f}>"
            
            x_right_toks = self.tokenizer.encode(string_value, return_tensors='pt')
            
            x_right_toks = x_right_toks.to(f'cuda:{self.left_model_device}')
            
            if cache_trajectory_pairs:
                dd = {}
                dd["full_input_left_IDL"] = x.to(f'cpu:{0}')
                dd["full_output_left_IDL"] = x_left.to(f'cpu:{0}')
                dd["full_input_right_IDL"] = x_right.to(f'cpu:{0}')
                dd["full_output_right_IDL"] = scalar_value
                dd["iterr"] = iterr_
                self.IDL_trajectory_pairs.append(dd)
            
            size_of_new_input = input_ids.shape[1] + x_left_only.shape[1] + x_right_toks.shape[1]
            
            if ( size_of_new_input>=self.max_internal_seq_length):
                x_left_only = x_left_only[:, :(x_left_only.shape[1]-(input_ids.shape[1]+x_right_toks.shape[1]) )] # lop off the last section  
                
            x = torch.cat((input_ids.to(f'cuda:{self.left_model_device}'), x_left_only.to(f'cuda:{self.left_model_device}'), x_right_toks.to(f'cuda:{self.left_model_device}')), dim=-1)
            iterr_ +=1
        
            if x_score > self.IDL_stopping_probability_thresh:
                break
            
        return x_left, scalar_value

    def tril_expand(self, toks):
        # input: toks as a list
        # output: expand in axis 0 and tril upper right
        seq_len = toks.shape[1]
        tril_matrix = torch.tril(torch.ones((seq_len, seq_len), device=toks.device))
        toks_expand = toks.expand(seq_len,seq_len)
        toks_expand_tril = (toks_expand * tril_matrix).long()
        return toks_expand_tril 
        
    def backprop_right_model(self, positive_response):
        # we want to form a minibatch of:
        # positives = tril(target_response)
        # negatives = full_output_left_IDLs
        
        loss_fn = nn.CrossEntropyLoss()
        
        max_microbatch_size = self.max_microbatch_size
        
        positive_response = self.add_special_tok_ids(positive_response)

        # full_output_left_IDL or full_input_right_IDL? I think full_input_right_IDL bc that is what will actually go into the model at test time? .. isnt it the same?
        negative_batch_lists = [self.add_special_tok_ids(i["full_input_right_IDL"]) for i in self.IDL_trajectory_pairs]
        
        ####
        # TODO: perhaps I should ONLY train on the "highest scoring" negative trajectories... (hard examples)
        ### 
        
        max_length = max(positive_response.shape[1], max([i.shape[1] for i in negative_batch_lists]))
        
        positive_response = positive_response.to(self.right_model_device)
        # Create a batch using tril() to generate a lower triangular matrix
        batch_size, seq_len = positive_response.size()
        tril_matrix = torch.tril(torch.ones((seq_len, seq_len), device=self.right_model_device))

        # Flatten target tokens for the loss computation
        positive_response_flat = positive_response.view(-1)
        positive_response_flat_expanded = positive_response_flat.expand(seq_len,seq_len)
        
        # Generate a batch of input sequences
        positive_batch = (positive_response_flat_expanded * tril_matrix).long()

        positive_batch_padded = nn.functional.pad(positive_batch, (0, max_length - seq_len), value=self.tokenizer.pad_token_id)
                          
        negative_batch_lists_padded = [self.tril_expand(nn.functional.pad(i, (0, max_length - i.size(1)), value=self.tokenizer.pad_token_id)) for i in negative_batch_lists]
        
        negative_batch_padded = torch.cat([i for i in negative_batch_lists_padded], dim=0).to(f'cuda:{self.right_model_device}')
        
        indices = torch.randperm(negative_batch_padded.shape[0])[:positive_batch_padded.shape[0]]

        sampled_negative_batch_padded = negative_batch_padded[indices]

        # Create targets
        positive_batch_targets = torch.ones(positive_batch_padded.shape[0], dtype=torch.float32).to(self.right_model_device)
        negative_batch_targets = torch.zeros(sampled_negative_batch_padded.shape[0], dtype=torch.float32).to(self.right_model_device)
        
        all_samples = torch.cat([positive_batch_padded,sampled_negative_batch_padded]).long()
        all_masks = (all_samples > 0).int()
        all_targets = torch.cat([positive_batch_targets,negative_batch_targets]).long()

        num_microbatches = (all_samples.size(0) + max_microbatch_size - 1) // max_microbatch_size
        
        # Initialize loss for accumulation
        total_loss = 0.0
        total_correct = 0 
        
        for i in range(num_microbatches):
            start_idx = i * max_microbatch_size
            end_idx = min(start_idx + max_microbatch_size, all_samples.size(0))
            
            # Get the microbatch
            microbatch = all_samples[start_idx:end_idx, :]
            microbatch_mask = all_masks[start_idx:end_idx, :]
            microbatch_targets = all_targets[start_idx:end_idx]
            
            # Forward pass through the right model and valence layer (11x462)
            outputs = self.valence_layer(self.right_model(microbatch.to(self.right_model_device), return_dict=True).logits)
            
            # Sum of each row
            row_sums = microbatch_mask.sum(dim=1, keepdim=True)
            
            # Avoid division by zero by setting zeros to 1 (or handle as appropriate)
            row_sums[row_sums == 0] = 1e-10
            
            # Normalize each row by dividing by the row sum
            normalized_microbatch_mask = microbatch_mask / row_sums

            expanded_mask = normalized_microbatch_mask.unsqueeze(-1).expand_as(outputs)
            outputs = outputs*expanded_mask #mask out padded values, average the rest.. we could do the pavlovian decay here.. but its ok for now

            outputs_summed = torch.sum(outputs,dim=1)  # Sum across seq_len

            microbatch_targets = microbatch_targets.to(outputs.device)

            # Calculate binary cross-entropy loss
            loss = loss_fn(outputs_summed, microbatch_targets)
            
            # Backpropagation
            loss.backward() #VRAM (left=1GB and right = 8.7GB)
            
            # Accumulate loss
            total_loss += loss.item()
    
            # Calculate accuracy
            total_correct += ((outputs_summed[:,1] > 0.5).float() == microbatch_targets).float().sum()
        
        accuracy=total_correct/all_samples.size(0)
        return total_loss, accuracy


    
    def backprop_step_right_model(self):
        nn.utils.clip_grad_norm_(self.right_model.parameters(), self.max_grad_norm)
        self.optimizer_right.step()
        self.optimizer_right.zero_grad()
    
    # def recompute_trajectory_valences():
    #     for i, (input_ids, x_left, _) in enumerate(self.IDL_trajectory_pairs):
    #         concatenated_input = torch.cat([input_ids, x_left], dim=-1)
    #         outputs_right = self.right_model(concatenated_input, return_dict=True)
    #         y_hat_right = self.valence_layer(outputs_right.logits)
    #         recomputed_valence = y_hat_right[0, -1, 0].item()
    #         self.IDL_trajectory_pairs[i] = (input_ids, x_left, y_hat_right)
    def calculate_policy_loss(self, looked_up_values,valences):
        # PPO style objective function
        surr1 = looked_up_values.mean() * valences
        surr2 = torch.clamp(looked_up_values.mean(), min=1.0 - self.epsilon, max=1.0 + self.epsilon) * valences
        policy_loss = -torch.min(surr1, surr2).mean()
        return policy_loss
        
    def backprop_left_model_policy(self, target_trajectory):
        if not self.IDL_trajectory_pairs:
            raise ValueError("IDL_trajectory_pairs is empty. Ensure it is populated before calling backprop_left_model.")
        
        # max_length = max(pair[1].size(1) for pair in self.IDL_trajectory_pairs)
        
        # padded_inputs = [nn.functional.pad(pair[1], (0, max_length - pair[1].size(1)), value=self.tokenizer.pad_token_id) for pair in self.IDL_trajectory_pairs]
        # inputs = torch.cat(padded_inputs)
    
        # Form a batch from IDL_trajectory_pairs and call self.right_model in one shot to compute the valences
        # batch_concatenated_inputs = [torch.cat([pair[0], pair[1]], dim=-1) for pair in self.IDL_trajectory_pairs]
        # max_concat_length = max(tensor.size(1) for tensor in batch_concatenated_inputs)
        # padded_concatenated_inputs = [nn.functional.pad(tensor, (0, max_concat_length - tensor.size(1)), value=self.tokenizer.pad_token_id) for tensor in batch_concatenated_inputs]
        # batch_inputs = torch.cat(padded_concatenated_inputs)
        target_trajectory = self.add_special_tok_ids(target_trajectory)
        
        batch_inputs = target_trajectory

        # get pi_thetas...
        logits = self.left_model(batch_inputs.to(f'cuda:{self.left_model_device}'), return_dict=True).logits

        log_probs = torch.log(F.softmax(logits, dim=-1))
        
        batch_inputs = batch_inputs.to(f'cuda:{self.left_model_device}')
    
        # Lookup the values
        looked_up_values = torch.gather(log_probs, 2, batch_inputs.unsqueeze(-1))
        looked_up_values = looked_up_values.squeeze(-1).to(f'cuda:{self.left_model_device}')  # Shape: [10, 490]
        
        # get valences r(tau)...
        outputs_right = self.right_model(batch_inputs.to(f'cuda:{self.right_model_device}'), return_dict=True)
        y_hat_right = self.valence_layer(outputs_right.logits).to(f'cuda:{self.right_model_device}')
        
        # valences = y_hat_right[:, -1, 0]  # Extract the relevant valences
        valences = y_hat_right.to(f'cuda:{self.left_model_device}') # batch, seq_len, 2
        positive_valence = (torch.mean(valences[:,:,1])-self.baseline_score).item()
        
        policy_loss = self.calculate_policy_loss(looked_up_values, positive_valence)

        policy_loss.backward() # TODO: should we backprop through to both models?? 
        
        # self.optimizer_left.zero_grad()
        # for dd in self.IDL_trajectory_pairs:
        #    policy_loss.backward()        
    
        average_valence = positive_valence
        

        #####
        # TODO: Need to add a negative sampling as well! 
        # loop over IDL_trajectories, and then backprop on that as well... 
        #####
        for dd in self.IDL_trajectory_pairs:
            negative_batch_trajectory = self.add_special_tok_ids(dd["full_output_left_IDL"])
            negative_valence = dd["full_output_right_IDL"]
            
            batch_inputs = negative_batch_trajectory
    
            # get pi_thetas...
            logits = self.left_model(batch_inputs.to(f'cuda:{self.left_model_device}'), return_dict=True).logits
    
            log_probs = torch.log(F.softmax(logits, dim=-1))
            
            batch_inputs = batch_inputs.to(f'cuda:{self.left_model_device}')
        
            # Lookup the values
            looked_up_values = torch.gather(log_probs, 2, batch_inputs.unsqueeze(-1))
            
            policy_loss = self.calculate_policy_loss(looked_up_values, negative_valence)

            policy_loss.backward()
            average_valence += negative_valence
            # print("breakkkinng")
            break 

        return policy_loss.item(), average_valence/(len(self.IDL_trajectory_pairs)+1)

    def backprop_left_model_Cross_Entropy(self, full, max_microbatch_size=20):
        # Ensure target tokens are in tensor format and move them to the correct device
        full = self.add_special_tok_ids(full)
        
        #shift right one
        target_tokens = full[:, 1:].to(f'cuda:{self.left_model_device}').contiguous()
        
        # Create a batch using tril() to generate a lower triangular matrix
        batch_size, seq_len = target_tokens.size()
        tril_matrix = torch.tril(torch.ones((seq_len, seq_len), device=self.left_model_device))

        # Flatten target tokens for the loss computation
        target_tokens_flat = target_tokens.view(-1)
    
        # Initialize the loss function
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # Use -100 to ignore padding tokens
        
        # Generate a batch of input sequences
        # input_batch = (full.unsqueeze(1) * tril_matrix.unsqueeze(0)).long()
        
        input_batch = (full[:,:-1].to('cuda:0') * tril_matrix.unsqueeze(0)).long()
        
        # Flatten the batch to feed into the model
        input_batch_flat = input_batch.view(-1, seq_len)
        num_microbatches = (input_batch_flat.size(0) + max_microbatch_size - 1) // max_microbatch_size
        
        # Initialize loss for accumulation
        total_loss = 0.0
        
        for i in range(num_microbatches):
            start_idx = i * max_microbatch_size
            end_idx = min(start_idx + max_microbatch_size, input_batch_flat.size(0))
            
            # Get the microbatch
            microbatch = input_batch_flat[start_idx:end_idx, :]
            
            # Forward pass through the left model
            outputs = self.left_model(microbatch, return_dict=True)
            shift_logits = outputs.logits  # Shape: (microbatch_size, seq_len, vocab_size)
            # shift_logits = logits[:, :-1, :].contiguous()

            
            # Mask target tokens for the current microbatch
            target_microbatch = target_tokens.unsqueeze(1).expand(-1, seq_len, -1).contiguous()
            mask = tril_matrix.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            target_microbatch = target_microbatch * mask + (-100) * (1 - mask)
            target_microbatch = target_microbatch.view(-1).long()
            # pdb.set_trace()
            
            # loss = loss_fn(logits, target_microbatch[:,start_idx:end_idx,:])
            
            # Compute the loss
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), target_microbatch[start_idx * seq_len: end_idx * seq_len])

            # Backpropagation
            loss.backward()
            
            # Accumulate loss
            total_loss += loss.item()
            
            
    
        return total_loss / num_microbatches


    # def backprop_left_model_Cross_Entropy(self, target_tokens, max_microbatch_size = 100):
    #     # Ensure target tokens are in tensor format and move them to the correct device
    #     if isinstance(target_tokens, dict):  # If input is from tokenizer
    #         target_tokens = target_tokens['input_ids']
    #     target_tokens = target_tokens.to(self.left_model_device)
    
    #     # Create a batch using tril() to generate a lower triangular matrix
    #     batch_size, seq_len = target_tokens.size()
    #     tril_matrix = torch.tril(torch.ones((seq_len, seq_len), device=self.left_model_device))
    #     # Flatten target tokens for the loss computation
    #     target_tokens_flat = target_tokens.view(-1)

    
    #     # Adjust the size of the target tokens to match the logits
    #     target_tokens_expanded = target_tokens_flat.unsqueeze(1).expand(-1, seq_len).contiguous().view(-1)

    #     # Compute Cross Entropy loss
    #     loss_fn = nn.CrossEntropyLoss()
    #      # Backpropagation
    #     self.optimizer_left.zero_grad()
        
    #     # Generate a batch of input sequences
    #     input_batch = (target_tokens.unsqueeze(1) * tril_matrix.unsqueeze(0)).long()
        
    #     pdb.set_trace()
    #     # Flatten the batch to feed into the model
    #     input_batch_flat = input_batch.view(-1, seq_len) 
    #     num_microbatches = (input_batch_flat.size(0) + max_microbatch_size - 1) // max_microbatch_size
    #     for i in range(num_microbatches):
    #         start_idx = i * max_microbatch_size
    #         end_idx = min(start_idx + max_microbatch_size, input_batch_flat.size(0))
            
    #         # Get the microbatch
    #         microbatch = input_batch_flat[start_idx:end_idx,:]
    #         # Forward pass through the left model
    #         outputs = self.left_model(microbatch, return_dict=True)
    #         logits = outputs.logits
    #         pdb.set_trace()
    #         loss = loss_fn(logits.permute(0, 2, 1), target_tokens_expanded)
    #         loss.backward()
        
    #     return loss.item()
    
    # Example usage:
    # model = StrangeLoopNetwork(base_model, tokenizer)
    # target_tokens = [...]  # Your target tokens here
    # loss = model.backprop_left_model_Cross_Entropy(target_tokens)
    # print(f"Cross Entropy Loss: {loss}")
    def backprop_step_left_model(self):
        nn.utils.clip_grad_norm_(self.left_model.parameters(), self.max_grad_norm)
        self.optimizer_left.step()
        self.optimizer_left.zero_grad()
        
    def add_special_toks_string(self,toks):
        return [self.tokenizer.bos_token + toks + self.tokenizer.eos_token]
    
    # def add_special_tok_ids(self, tok_ids, bos_only=False):
        
    #     # Check if the first token is the BOS token
    #     if tok_ids[0, 0] != self.tokenizer.bos_token_id:
    #         tok_ids = torch.cat((torch.tensor([[self.tokenizer.bos_token_id]], device=tok_ids.device), tok_ids), dim=1)
        
    #     if bos_only:
    #         return tok_ids
            
    #     # Check if the last token is the EOS token
    #     if tok_ids[0, -1] != self.tokenizer.eos_token_id:
    #         tok_ids = torch.cat((tok_ids, torch.tensor([[self.tokenizer.eos_token_id]], device=tok_ids.device)), dim=1)
        
    #     return tok_ids
    def add_special_tok_ids(self, tok_ids, bos_only=False):
        # Flatten the tok_ids for searching and removing BOS and EOS tokens
        tok_ids_flat = tok_ids.view(-1).tolist()
    
        # Remove all instances of BOS and EOS tokens
        tok_ids_filtered = [token for token in tok_ids_flat if token not in [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]]
    
        # Convert back to tensor
        tok_ids = torch.tensor(tok_ids_filtered, device=tok_ids.device).unsqueeze(0)
    
        # Add BOS token at the beginning if not already present
        if tok_ids[0, 0] != self.tokenizer.bos_token_id:
            tok_ids = torch.cat((torch.tensor([[self.tokenizer.bos_token_id]], device=tok_ids.device), tok_ids), dim=1)
    
        if bos_only:
            return tok_ids
    
        # Add EOS token at the end if not already present
        if tok_ids[0, -1] != self.tokenizer.eos_token_id:
            tok_ids = torch.cat((tok_ids, torch.tensor([[self.tokenizer.eos_token_id]], device=tok_ids.device)), dim=1)
    
        return tok_ids
        
    def cut_all_after_eos_token(self, tok_ids):
        eos_token_id = self.tokenizer.eos_token_id
        # Find the index of the first occurrence of the EOS token
        if eos_token_id in tok_ids:
            eos_index = (tok_ids == eos_token_id).nonzero(as_tuple=True)[0][0]
            # Cut off all tokens after the EOS token
            return tok_ids[:eos_index + 1]
        else:
            # If EOS token is not found, return the original tok_ids
            return tok_ids



