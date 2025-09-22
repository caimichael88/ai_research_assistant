
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers/PEFT not available. Install with: pip install transformers peft")
    exit(1)
import torch
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

class LLMService:

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_loaded = False

    def load_trained_model(self, model_dir="../../models/hybrid_apple_model"):
        """load trained model and tokenizer"""
        print(f"load model form {model_dir}")

        try:
            base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
            print(f"Loading base model {base_model_name}")

            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Get the absolute path to the adapter directory
            current_file = Path(__file__).resolve()  # llm_service.py
            api_py_root = current_file.parent.parent.parent  # apps/api-py/
            adapter_path = api_py_root / "models" / "hybrid_apple_model"

            if adapter_path.exists():
                print(f"Loading PEFT adapter from {adapter_path}")
                model = PeftModel.from_pretrained(model, adapter_path)
            else:
                print(f"No adapter found at {adapter_path}, using base model")
            
            #set device
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else  "cpu"
            print(f"Using device {device}")

            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.is_loaded = True
            return model, tokenizer, device
    
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
        
    def model_inference(self, user_text):

        print(f"Model Inference {user_text}")

        if self.model is None:
            raise Exception("Model failed to load. Check if model exists and model files are valid")
        
        self.model.eval()

        try:
            message = [{"role": "user", "content": user_text}]

            inputs = self.tokenizer.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.device)

            attention_mask= torch.ones_like(inputs).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id= self.tokenizer.eos_token_id
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Response: {response}")

                #Extract just the generated part (after the prompt)
                input_length = len(self.tokenizer.decode(inputs[0], skip_special_tokens=True))
                generated_response = response[input_length:].strip()

                print(f"generated response for {user_text}: {generated_response}")
                return generated_response
        
        except Exception as e:
            print(f"Error doing inference with {user_text}: {e}")
            return f"Error during model inference: {str(e)}"

# Global service instance
llm_service = LLMService()