"""
LLM module: Groq (fast, free tier) or Local (fine-tuned model)
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import logging
from datetime import datetime

from config import get_settings, NUDGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def generate(
        self,
        user_message: str,
        memory_context: str = "",
        conversation_history: str = "",
        **kwargs
    ) -> str:
        """Generate a response from the LLM"""
        pass
    
    def build_prompt(
        self,
        user_message: str,
        memory_context: str = "",
        conversation_history: str = ""
    ) -> str:
        """
        Build the full prompt with memory injection
        
        This is the core of the strategy:
        - Inject long-term memory into system prompt
        - Include recent conversation history
        - Use the Nudge system prompt with identity shifting
        """
        today_date = datetime.now().strftime("%A, %B %d, %Y")
        
        # Format memory context for the prompt
        memory_text = memory_context if memory_context else "(No memories yet - this is a new user)"
        
        # Build system prompt with placeholders filled
        system_prompt = NUDGE_SYSTEM_PROMPT.format(
            today_date=today_date,
            memory_context=memory_text
        )
        
        # Build user context
        prompt_parts = []
        
        # Add conversation history if available
        if conversation_history:
            prompt_parts.append(f"Recent conversation:\n{conversation_history}")
        
        # Add current message
        prompt_parts.append(f"User: {user_message}")
        
        # Join all parts
        full_context = "\n\n".join(prompt_parts)
        
        return system_prompt, full_context


class GroqLLM(BaseLLM):
    """
    Groq-hosted Llama 3.1 (free tier: 500+ tok/s, 1M+ tokens/month)
    Best for production - fast, free, no infrastructure needed
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._client = None
    
    @property
    def client(self):  
        """Lazy load the Groq client"""
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.settings.groq_api_key)
            except ImportError:
                raise ImportError("Please install groq: pip install groq")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Groq client: {e}")
        return self._client
    
    def generate(
        self,
        user_message: str,
        memory_context: str = "",
        conversation_history: str = "",
        **kwargs
    ) -> str:
        """Generate response using Groq API"""
        
        system_prompt, full_context = self.build_prompt(
            user_message, memory_context, conversation_history
        )
        
        # Build messages for chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_context}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.settings.groq_model,
                messages=messages,
                temperature=0.3,   # Balanced: rule adherence + variety/wit
                max_tokens=250,    # Enough for complete actions + context
                top_p=0.8          # Balanced variety
                # Note: Groq API doesn't support repetition_penalty parameter
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            # Return a fallback response instead of raising
            return "Try again—connection hiccup. Done? Yes/No"


class LocalLLM(BaseLLM):
    """
    Self-hosted fine-tuned Llama 3.1 with LoRA adapter
    Use this if you've fine-tuned the model with the Nudge personality
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Load the fine-tuned model"""
        if self._model is not None:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            logger.info(f"Loading base model: {self.settings.local_base_model}")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.settings.local_base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
            )
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.settings.local_base_model
            )
            
            # Load and merge LoRA adapter if it exists
            import os
            if os.path.exists(self.settings.local_model_path):
                logger.info(f"Loading LoRA adapter: {self.settings.local_model_path}")
                self._model = PeftModel.from_pretrained(
                    base_model,
                    self.settings.local_model_path
                )
                # Merge for faster inference
                self._model = self._model.merge_and_unload()
            else:
                logger.warning(f"LoRA adapter not found at {self.settings.local_model_path}, using base model")
                self._model = base_model
            
            self._model.eval()
            logger.info("Model loaded successfully!")
            
        except ImportError as e:
            raise ImportError(f"Please install transformers and peft: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def generate(
        self,
        user_message: str,
        memory_context: str = "",
        conversation_history: str = "",
        **kwargs
    ) -> str:
        """Generate response using local model"""
        import torch
        
        system_prompt, full_context = self.build_prompt(
            user_message, memory_context, conversation_history
        )
        
        # Build Llama 3.1 chat format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{full_context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        # Decode and extract response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract just the assistant's response
        response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        response = response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
        
        return response


class MockLLM(BaseLLM):
    """Mock LLM for testing without API keys - Unlimits style"""
    
    def generate(
        self,
        user_message: str,
        memory_context: str = "",
        conversation_history: str = "",
        **kwargs
    ) -> str:
        """Return a mock Nudge-style response - executable actions only"""
        
        # Always respond in clean English (no Hinglish)
        msg = user_message.lower()
        
        if "tired" in msg or "burnt" in msg or "exhausted" in msg:
            return """I hear you. Here's your executable action right now:

Stand up. Walk to the nearest window. Look outside for 60 seconds. Then come back and open exactly ONE file related to your dream project.

Do that file walkthrough now? Yes/No"""
        
        if "stuck" in msg or "confused" in msg or "lost" in msg:
            return """Here's your 10-minute action to break the stuck feeling:

Open your project folder. Find the smallest incomplete function or task. Write exactly 5 lines of code or content. Ship it.

Starting that right now? Yes/No"""
        
        # Default: extract their dream
        return """I'm Nudge, the Unlimits Achievement Coach. I turn your bold dream into daily reality through concrete micro-actions.

Tell me: What's the ONE dream you're chasing? Be specific — what exactly do you want to achieve and by when?

What's your dream?"""


# Factory function to get the appropriate LLM
_llm_instance: Optional[BaseLLM] = None


def get_llm() -> BaseLLM:
    """Get the configured LLM instance"""
    global _llm_instance
    
    if _llm_instance is not None:
        return _llm_instance
    
    settings = get_settings()
    
    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY not set, using mock LLM")
            _llm_instance = MockLLM()
        else:
            _llm_instance = GroqLLM()
            
    elif settings.llm_provider == "local":
        _llm_instance = LocalLLM()
        
    else:
        logger.warning(f"Unknown LLM provider: {settings.llm_provider}, using mock")
        _llm_instance = MockLLM()
    
    return _llm_instance

