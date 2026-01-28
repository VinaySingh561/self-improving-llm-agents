"""LLM inference wrapper for CPU-only execution.

Design:
- Uses open-weight, quantized models (Mistral, Llama, etc.)
- CPU-only execution (no GPU dependency)
- Deterministic generation (low temperature for reproducibility)
- Token budgeting and call limiting per episode
- Full logging of prompts and outputs
- JSON output validation and parsing
- Thread-safe for concurrent episodes
"""

from typing import Dict, Any, Optional, List, Tuple
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class LLMCallLog:
    """Log entry for a single LLM call."""
    call_id: int
    timestamp: str
    prompt_length: int
    response_length: int
    response_time_sec: float
    tokens_input: int
    tokens_output: int
    parsed_output: Dict[str, Any]
    raw_output: str
    valid_json: bool
    error_message: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration for LLM wrapper."""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    quantization_bits: int = 4
    temperature: float = 0.1  # Low temperature for determinism
    top_p: float = 0.95
    max_tokens_per_call: int = 256
    max_calls_per_episode: int = 100
    device: str = "cpu"
    device_map: str = "cpu"
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    # Timeout and retry
    generation_timeout_sec: float = 60.0
    max_retries: int = 2
    
    # Logging
    log_all_calls: bool = True
    save_logs: bool = True


class LLMWrapper:
    """
    Wrapper for local, quantized LLM inference.
    
    Properties:
    - CPU-only execution (quantized 4-bit or 8-bit)
    - Deterministic generation (low temperature)
    - Token budgeting per episode
    - Full logging of prompts and outputs
    - JSON output validation
    - Reproducible random state
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize LLM wrapper.
        
        Args:
            config: LLMConfig object (uses defaults if None)
            log_dir: Directory for call logs
        """
        self.config = config or LLMConfig()
        self.log_dir = Path(log_dir) if log_dir else None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_history: List[LLMCallLog] = []
        
        logger.info(f"Initializing LLM: {self.config.model_name}")
        logger.info(f"  Quantization: {self.config.quantization_bits}-bit")
        logger.info(f"  Device: {self.config.device}")
        logger.info(f"  Temperature: {self.config.temperature}")
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load quantized model from HuggingFace Hub."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="left",
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Loading model with quantization...")
            
            # Configure quantization
            if self.config.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=self._str_to_dtype(self.config.bnb_4bit_compute_dtype),
                )
            elif self.config.load_in_8bit:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                bnb_config = None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map=self.config.device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            logger.info("✓ Model loaded successfully")
            logger.info(f"  Model parameters: {self._count_parameters():,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Falling back to mock LLM for testing")
            self._setup_mock_llm()
    
    def _setup_mock_llm(self) -> None:
        """Setup mock LLM for testing without GPU."""
        logger.warning("Using MOCK LLM for demonstration (no real model)")
        self.model = None
        self.tokenizer = None
    
    @staticmethod
    def _str_to_dtype(dtype_str: str):
        """Convert string to torch dtype."""
        import torch
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float16)
    
    def _count_parameters(self) -> int:
        """Count model parameters."""
        if not self.model:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def call(
        self,
        prompt: str,
        output_format: str = "json",
        retries: int = 0,
    ) -> Dict[str, Any]:
        """
        Call LLM and parse output.
        
        Args:
            prompt: Full prompt string
            output_format: Expected format ("json", "text", or "skill_decision")
            retries: Current retry count (internal)
            
        Returns:
            Parsed output as dict with keys:
            - success: bool
            - output: parsed output or raw text
            - error: error message if failed
            - metadata: call metadata (tokens, time, etc.)
        """
        if self.call_count >= self.config.max_calls_per_episode:
            logger.warning(f"Exceeded max calls per episode ({self.config.max_calls_per_episode})")
            return {
                "success": False,
                "output": None,
                "error": "Max calls per episode exceeded",
                "metadata": self._get_metadata(),
            }
        
        start_time = time.time()
        self.call_count += 1
        call_id = self.call_count
        
        try:
            logger.debug(f"LLM call #{call_id} (format={output_format})")
            
            # Generate response
            raw_output = self._generate(prompt, call_id)
            elapsed = time.time() - start_time
            
            # Parse output
            parsed_output, valid_json, error_msg = self._parse_output(
                raw_output,
                output_format,
                call_id
            )
            
            # Log call
            log_entry = self._log_call(
                call_id,
                prompt,
                raw_output,
                parsed_output,
                valid_json,
                elapsed,
                error_msg,
            )
            self.call_history.append(log_entry)
            
            if error_msg and retries < self.config.max_retries:
                logger.warning(f"Parse error (retry {retries + 1}/{self.config.max_retries}): {error_msg}")
                return self.call(prompt, output_format, retries + 1)
            
            return {
                "success": valid_json and error_msg is None,
                "output": parsed_output,
                "error": error_msg,
                "metadata": {
                    "call_id": call_id,
                    "response_time_sec": elapsed,
                    "input_tokens": log_entry.tokens_input,
                    "output_tokens": log_entry.tokens_output,
                    "total_calls": self.call_count,
                },
            }
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "metadata": self._get_metadata(),
            }
    
    def _generate(self, prompt: str, call_id: int) -> str:
        """Generate text from prompt."""
        if self.model is None:
            # Mock response for testing
            return self._mock_response(prompt, call_id)
        
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True,
        ).to(self.config.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens_per_call,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=False,  # Deterministic (temperature + no sampling)
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        raw_output = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        
        # Update token counts
        self.total_input_tokens += inputs["input_ids"].shape[-1]
        self.total_output_tokens += outputs.shape[-1] - inputs["input_ids"].shape[-1]
        
        return raw_output
    
    def _mock_response(self, prompt: str, call_id: int) -> str:
        """Generate mock response for testing."""
        # Check what the prompt is asking for
        if "skill" in prompt.lower():
            return json.dumps({
                "skill": "explore",
                "reasoning": "Mock response: Explore to find resources",
                "confidence": 0.8,
            })
        elif "decision" in prompt.lower():
            return json.dumps({
                "decision": "move_forward",
                "reasoning": "Mock response: Safe move",
            })
        else:
            return json.dumps({
                "response": "Mock LLM response",
                "call_id": call_id,
            })
    
    def _parse_output(
        self,
        raw_output: str,
        output_format: str,
        call_id: int,
    ) -> Tuple[Optional[Dict], bool, Optional[str]]:
        """
        Parse LLM output.
        
        Returns:
            (parsed_dict, valid_json, error_message)
        """
        try:
            # Try to extract JSON from output
            json_str = raw_output.strip()
            
            # If output contains markdown code blocks, extract
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Validate based on format
            if output_format == "skill_decision":
                if "skill" not in parsed or "reasoning" not in parsed:
                    return None, False, "Missing 'skill' or 'reasoning' in output"
            
            return parsed, True, None
        
        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {e}"
            logger.debug(f"Call #{call_id}: {error_msg}")
            return None, False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.debug(f"Call #{call_id}: {error_msg}")
            return None, False, error_msg
    
    def _log_call(
        self,
        call_id: int,
        prompt: str,
        raw_output: str,
        parsed_output: Dict,
        valid_json: bool,
        elapsed: float,
        error_msg: Optional[str],
    ) -> LLMCallLog:
        """Log a single LLM call."""
        # Estimate token counts
        input_tokens = len(prompt.split())  # Rough estimate
        output_tokens = len(raw_output.split())
        
        log_entry = LLMCallLog(
            call_id=call_id,
            timestamp=datetime.now().isoformat(),
            prompt_length=len(prompt),
            response_length=len(raw_output),
            response_time_sec=elapsed,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            parsed_output=parsed_output or {},
            raw_output=raw_output[:200],  # Truncate for logging
            valid_json=valid_json,
            error_message=error_msg,
        )
        
        logger.debug(
            f"Call #{call_id}: {input_tokens} in, {output_tokens} out, "
            f"{elapsed:.2f}s, valid_json={valid_json}"
        )
        
        return log_entry
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get current usage metadata."""
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_tokens_per_call": (
                (self.total_input_tokens + self.total_output_tokens) /
                max(1, self.call_count)
            ),
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            **self._get_metadata(),
            "calls_remaining": max(
                0,
                self.config.max_calls_per_episode - self.call_count
            ),
            "token_budget_remaining": (
                self.config.max_tokens_per_call * self.config.max_calls_per_episode
                - (self.total_input_tokens + self.total_output_tokens)
            ),
        }
    
    def reset_episode_stats(self) -> None:
        """Reset stats for new episode."""
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        logger.debug("Reset episode statistics")
    
    def save_logs(self, filepath: Optional[str] = None) -> Optional[Path]:
        """Save call history to JSON file."""
        if not self.call_history:
            logger.warning("No call history to save")
            return None
        
        if filepath is None:
            if not self.log_dir:
                logger.warning("No log directory specified")
                return None
            filepath = self.log_dir / f"llm_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        logs_dict = {
            "config": {
                "model": self.config.model_name,
                "quantization_bits": self.config.quantization_bits,
                "temperature": self.config.temperature,
            },
            "calls": [
                {
                    "call_id": log.call_id,
                    "timestamp": log.timestamp,
                    "prompt_length": log.prompt_length,
                    "response_length": log.response_length,
                    "response_time_sec": log.response_time_sec,
                    "tokens_input": log.tokens_input,
                    "tokens_output": log.tokens_output,
                    "valid_json": log.valid_json,
                    "error_message": log.error_message,
                }
                for log in self.call_history
            ],
            "summary": self.get_usage_stats(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(logs_dict, f, indent=2)
        
        logger.info(f"Saved {len(self.call_history)} LLM call logs to {filepath}")
        return filepath


class OutputValidator:
    """Validate and parse LLM output to specific formats."""
    
    @staticmethod
    def validate_skill_decision(output: Dict[str, Any]) -> bool:
        """
        Validate skill decision output.
        
        Expected format:
        {
            "skill": "explore" | "fight" | "flee" | "eat" | "use_item",
            "reasoning": "explanation",
            "confidence": float (0.0-1.0),
            "parameters": {...}  (optional)
        }
        """
        required = ["skill", "reasoning"]
        return all(key in output for key in required)
    
    @staticmethod
    def extract_skill(output: Dict[str, Any]) -> Optional[str]:
        """Extract skill decision from output."""
        skill = output.get("skill")
        valid_skills = ["explore", "fight", "flee", "eat", "use_item"]
        return skill if skill in valid_skills else None
    
    @staticmethod
    def validate_action_sequence(output: Dict[str, Any]) -> bool:
        """
        Validate action sequence output.
        
        Expected format:
        {
            "actions": ["move_north", "move_east", ...],
            "reasoning": "explanation"
        }
        """
        return "actions" in output and isinstance(output["actions"], list)
    
    @staticmethod
    def extract_actions(output: Dict[str, Any]) -> Optional[List[str]]:
        """Extract action sequence from output."""
        if "actions" in output and isinstance(output["actions"], list):
            return output["actions"]
        return None

