from transformers import AutoTokenizer

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"  # Reset to default color
import pdb
def print_colored_text(valence_mask, token_ids, tokenizer):
    """
    Prints tokenized text with background colors based on valence_mask.
    
    Parameters:
    valence_mask (list of int): List of 0s and 1s indicating color (0 for red, 1 for green).
    token_ids (list of int): List of token IDs.
    tokenizer (object): Tokenizer with a decode method to decode token IDs.
    """
    
    for valence, token_id in zip(valence_mask, token_ids[0]):
        # Decode the token ID
        token = tokenizer.decode(token_id)
        
        # Determine the color based on valence_mask
        color = RED if valence  <= 0.5  else GREEN
        
        # Print the token with the appropriate background color
        print(f"{color}{token}{RESET}", end='')
    
    # Ensure the final reset of color
    print(RESET)

# Example usage (replace with actual valence_mask, token_ids, and tokenizer)
if __name__ == "__main__":
    valence_mask = [0, 1, 0, 1,0,1,0, 1, 0, 1,0,1,0, 1, 0, 1,0,1,0, 1, 0, 1,0,1,0, 1, 0, 1,0,1,0]
    token_ids = [0, 1, 2, 3,4,5,6,7,7,8,9,10,11,12,23,43,54,56,67,67,768,65,76,3534,654,7655,2433]
    base_model_id = "EleutherAI/pythia-410M"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    print_colored_text(valence_mask, token_ids, tokenizer)
