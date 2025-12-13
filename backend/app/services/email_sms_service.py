import torch
from transformers import DistilBertTokenizer
import os


class EmailSMSPhishingDetector:
    """Service for email/SMS phishing detection using DistilBERT"""

    def __init__(self, model_path, max_length=128):
        """
        Initialize the detector

        Args:
            model_path: Path to TorchScript model (.pt file)
            max_length: Maximum token sequence length
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length

        # Load tokenizer (downloads from HuggingFace on first run)
        print(f"Loading tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # Load TorchScript model
        print(f"Loading model from {model_path}...")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Optimize for inference
        if hasattr(torch.jit, 'optimize_for_inference'):
            self.model = torch.jit.optimize_for_inference(self.model)

        # Set threads for CPU inference
        if self.device.type == 'cpu':
            torch.set_num_threads(4)

        print(f"Model loaded successfully on {self.device}")

    def warmup(self):
        """Warm up model with dummy prediction to avoid cold start"""
        dummy_text = "This is a warmup message to initialize the model."
        _ = self.predict(dummy_text)
        print("Model warmed up")

    def predict(self, text):
        """
        Predict phishing score for given text

        Args:
            text: Email/SMS content (string)

        Returns:
            float: Phishing score (0-10)
        """
        # Tokenize input
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Get prediction
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probability = torch.sigmoid(logits).item()

        # Convert probability to score (0-10)
        return round(probability * 10, 2)


# Global model instance (loaded once on startup)
_detector = None


def get_detector():
    """Get or create the global detector instance"""
    global _detector

    if _detector is None:
        # Get model path from config
        from config import Config
        model_path = os.path.join(
            Config.MODEL_DIR,
            'email-sms-detection',
            'distillBERT_torchscript.pt'
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        _detector = EmailSMSPhishingDetector(model_path)
        _detector.warmup()

    return _detector
