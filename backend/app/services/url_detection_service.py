import torch
import pickle
import os
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np


class URLPhishingDetector:
    """Service for URL phishing detection using dual model architecture"""

    def __init__(self, cnn_model_path, dual_cnn_model_path, char_tokenizer_path,
                 char_tokenizer_cross_path, html_tokenizer_path, cnn_threshold_path,
                 dual_cnn_threshold_path, html_fetch_timeout=3):
        """
        Initialize the detector with both models

        Args:
            cnn_model_path: Path to CNN (URL-only) TorchScript model
            dual_cnn_model_path: Path to Dual CNN (URL+HTML) TorchScript model
            char_tokenizer_path: Path to character tokenizer pickle (for CNN)
            char_tokenizer_cross_path: Path to character tokenizer pickle (for cross-attention)
            html_tokenizer_path: Path to HTML word tokenizer pickle
            cnn_threshold_path: Path to CNN optimal threshold JSON
            dual_cnn_threshold_path: Path to Dual CNN optimal threshold JSON
            html_fetch_timeout: Timeout in seconds for HTML fetching (default: 3)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.html_fetch_timeout = html_fetch_timeout

        # Load character tokenizer for CNN (URL-only)
        print(f"Loading character tokenizer for CNN...")
        with open(char_tokenizer_path, 'rb') as f:
            char_tokenizer = pickle.load(f)
        self.char_to_idx_cnn = char_tokenizer['char_to_idx']
        self.max_url_length = char_tokenizer['max_length']

        # Load character tokenizer for cross-attention
        print(f"Loading character tokenizer for cross-attention...")
        with open(char_tokenizer_cross_path, 'rb') as f:
            char_tokenizer_cross = pickle.load(f)
        self.char_to_idx_cross = char_tokenizer_cross['char_to_idx']

        # Load HTML word tokenizer
        print(f"Loading HTML tokenizer...")
        with open(html_tokenizer_path, 'rb') as f:
            html_tokenizer = pickle.load(f)
        self.html_word_to_idx = html_tokenizer['word_to_idx']
        self.max_html_length = html_tokenizer['max_length']

        # Load optimal thresholds
        import json
        with open(cnn_threshold_path, 'r') as f:
            cnn_config = json.load(f)
        self.cnn_threshold = cnn_config['threshold']

        with open(dual_cnn_threshold_path, 'r') as f:
            dual_cnn_config = json.load(f)
        self.dual_cnn_threshold = dual_cnn_config['threshold']

        # Load CNN model (URL-only)
        print(f"Loading CNN model (URL-only)...")
        self.cnn_model = torch.jit.load(cnn_model_path, map_location=self.device)
        self.cnn_model.eval()

        # Load Dual CNN model (URL+HTML)
        print(f"Loading Dual CNN model (URL+HTML)...")
        self.dual_cnn_model = torch.jit.load(dual_cnn_model_path, map_location=self.device)
        self.dual_cnn_model.eval()

        # Optimize for inference
        if hasattr(torch.jit, 'optimize_for_inference'):
            self.cnn_model = torch.jit.optimize_for_inference(self.cnn_model)
            self.dual_cnn_model = torch.jit.optimize_for_inference(self.dual_cnn_model)

        # Set threads for CPU inference
        if self.device.type == 'cpu':
            torch.set_num_threads(4)

        print(f"Models loaded successfully on {self.device}")

    def warmup(self):
        """Warm up both models with dummy predictions"""
        dummy_url = "https://example.com"
        _ = self.predict(dummy_url)
        print("Models warmed up")

    def tokenize_url_cnn(self, url):
        """Convert URL to sequence of character indices (for CNN model)"""
        tokens = [self.char_to_idx_cnn.get(char, 0) for char in url[:self.max_url_length]]
        if len(tokens) < self.max_url_length:
            tokens += [0] * (self.max_url_length - len(tokens))
        return tokens

    def tokenize_url_cross(self, url):
        """Convert URL to sequence of character indices (for cross-attention model)"""
        tokens = [self.char_to_idx_cross.get(char, 0) for char in url[:self.max_url_length]]
        if len(tokens) < self.max_url_length:
            tokens += [0] * (self.max_url_length - len(tokens))
        return tokens

    def clean_html(self, html):
        """Clean HTML content for tokenization"""
        if not html or len(html) == 0:
            return ""

        try:
            # Truncate if too large
            if len(html) > 50000:
                html = html[:50000]

            soup = BeautifulSoup(html, 'lxml')

            # Remove unnecessary tags
            for tag in soup(['script', 'style', 'noscript', 'svg']):
                tag.decompose()

            # Extract text
            text = soup.get_text(separator=' ', strip=True)

            # Extract form actions
            forms = soup.find_all('form')[:3]
            form_actions = ' '.join([form.get('action', '')[:50] for form in forms])

            # Extract link hrefs
            links = soup.find_all('a')[:5]
            link_hrefs = ' '.join([link.get('href', '')[:30] for link in links])

            # Extract title
            title = soup.title.string if soup.title else ""
            if title and len(title) > 100:
                title = title[:100]

            # Combine and clean
            combined = f"{title} {text[:5000]} {form_actions} {link_hrefs}"
            combined = re.sub(r'\s+', ' ', combined).strip()

            return combined[:10000]

        except Exception as e:
            return ""

    def tokenize_html(self, text):
        """Convert HTML text to sequence of word indices"""
        words = text.lower().split()[:self.max_html_length]
        tokens = [self.html_word_to_idx.get(word, 1) for word in words]  # 1 = <UNK>

        # Pad to max_length
        if len(tokens) < self.max_html_length:
            tokens += [0] * (self.max_html_length - len(tokens))

        return tokens

    def fetch_html(self, url):
        """
        Fetch HTML content from URL with timeout

        Returns:
            str: HTML content or None if failed/timeout
        """
        try:
            response = requests.get(
                url,
                timeout=self.html_fetch_timeout,
                headers={'User-Agent': 'Mozilla/5.0'},
                allow_redirects=True
            )
            response.raise_for_status()
            return response.text
        except Exception:
            return None

    def predict_url_only(self, url):
        """Predict using CNN model (URL-only)"""
        # Tokenize URL with CNN tokenizer
        url_tokens = self.tokenize_url_cnn(url)
        url_tensor = torch.LongTensor([url_tokens]).to(self.device)

        # Get prediction
        with torch.no_grad():
            # CNN model returns (logits, attention_weights) tuple
            output = self.cnn_model(url_tensor)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            probability = torch.sigmoid(logits).item()

        # Convert probability to score (0-10)
        return round(probability * 10, 2)

    def predict_url_html(self, url, html):
        """Predict using Dual CNN model (URL+HTML)"""
        # Tokenize URL with cross-attention tokenizer
        url_tokens = self.tokenize_url_cross(url)
        url_tensor = torch.LongTensor([url_tokens]).to(self.device)

        # Clean and tokenize HTML
        cleaned_html = self.clean_html(html)
        html_tokens = self.tokenize_html(cleaned_html)
        html_tensor = torch.LongTensor([html_tokens]).to(self.device)

        # Get prediction
        with torch.no_grad():
            logits = self.dual_cnn_model(url_tensor, html_tensor)
            probability = torch.sigmoid(logits).item()

        # Convert probability to score (0-10)
        return round(probability * 10, 2)

    def predict(self, url):
        """
        Predict phishing score for URL using URL-only CNN model

        Args:
            url: URL string

        Returns:
            dict: {
                'score': float (0-10),
                'model_used': str ('url_only')
            }
        """
        # Use URL-only CNN model
        score = self.predict_url_only(url)
        return {
            'score': score,
            'model_used': 'url_only'
        }

        # ==========================================
        # CROSS-ATTENTION MODEL (COMMENTED OUT)
        # To enable: uncomment the code below
        # ==========================================
        # # Fetch HTML
        # html = self.fetch_html(url)

        # if html:
        #     # HTML fetch succeeded, use dual CNN model
        #     score = self.predict_url_html(url, html)
        #     return {
        #         'score': score,
        #         'model_used': 'cross_attention'
        #     }
        # else:
        #     # HTML fetch failed, fallback to URL-only
        #     score = self.predict_url_only(url)
        #     return {
        #         'score': score,
        #         'model_used': 'url_only'
        #     }


# Global model instance (loaded once on startup)
_detector = None


def get_detector():
    """Get or create the global detector instance"""
    global _detector

    if _detector is None:
        # Get paths from config
        from config import Config

        model_dir = Config.MODEL_DIR
        url_detection_dir = os.path.join(model_dir, 'url-detection')

        cnn_model_path = os.path.join(url_detection_dir, 'cnn_torchscript.pt')
        dual_cnn_model_path = os.path.join(url_detection_dir, 'dual_cnn_cross_attention.pt')
        char_tokenizer_path = os.path.join(url_detection_dir, 'char_tokenizer.pkl')
        char_tokenizer_cross_path = os.path.join(url_detection_dir, 'char_tokenizer_cross_attention.pkl')
        html_tokenizer_path = os.path.join(url_detection_dir, 'html_word_tokenizer.pkl')
        cnn_threshold_path = os.path.join(url_detection_dir, 'cnn_url_optimal_threshold.json')
        dual_cnn_threshold_path = os.path.join(url_detection_dir, 'dual_cnn_cross_attention_optimal_threshold.json')

        # Check if all files exist
        required_files = [
            cnn_model_path, dual_cnn_model_path, char_tokenizer_path,
            char_tokenizer_cross_path, html_tokenizer_path, cnn_threshold_path,
            dual_cnn_threshold_path
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        _detector = URLPhishingDetector(
            cnn_model_path=cnn_model_path,
            dual_cnn_model_path=dual_cnn_model_path,
            char_tokenizer_path=char_tokenizer_path,
            char_tokenizer_cross_path=char_tokenizer_cross_path,
            html_tokenizer_path=html_tokenizer_path,
            cnn_threshold_path=cnn_threshold_path,
            dual_cnn_threshold_path=dual_cnn_threshold_path,
            html_fetch_timeout=3
        )
        _detector.warmup()

    return _detector
