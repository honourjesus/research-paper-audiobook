import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from jiwer import wer, cer
from rouge import Rouge
from bert_score import score
import logging

class ModelEvaluator:
    """Evaluates model performance using multiple metrics"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.rouge = Rouge()
    
    def evaluate_paper_to_speech(self, 
                                 original_text: str, 
                                 generated_text: str,
                                 audio_file: str) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the paper-to-speech conversion
        
        Metrics:
        - Text preservation (ROUGE, BLEU)
        - Equation accuracy
        - Audio quality
        - Structure preservation
        - User satisfaction proxy
        """
        
        metrics = {}
        
        # 1. Text Preservation Metrics
        metrics['text_preservation'] = self._evaluate_text_preservation(
            original_text, generated_text
        )
        
        # 2. Equation Handling Accuracy
        metrics['equation_accuracy'] = self._evaluate_equations(
            original_text, generated_text
        )
        
        # 3. Structure Preservation
        metrics['structure_preservation'] = self._evaluate_structure(
            original_text, generated_text
        )
        
        # 4. Audio Quality (if audio provided)
        if audio_file:
            metrics['audio_quality'] = self._evaluate_audio_quality(audio_file)
        
        # 5. Content Coverage
        metrics['content_coverage'] = self._evaluate_coverage(
            original_text, generated_text
        )
        
        # Calculate overall score
        metrics['overall_score'] = self._calculate_overall_score(metrics)
        
        return metrics
    
    def _evaluate_text_preservation(self, original: str, generated: str) -> Dict:
        """Evaluate how well the text content is preserved"""
        
        # Word Error Rate
        word_error = wer(original, generated)
        
        # Character Error Rate
        char_error = cer(original, generated)
        
        # ROUGE scores
        try:
            rouge_scores = self.rouge.get_scores(generated, original)[0]
        except:
            rouge_scores = {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}
        
        # BERTScore (semantic similarity)
        P, R, F1 = score([generated], [original], lang="en", verbose=False)
        
        return {
            'wer': word_error,
            'cer': char_error,
            'rouge_1_f1': rouge_scores['rouge-1']['f'],
            'rouge_2_f1': rouge_scores['rouge-2']['f'],
            'rouge_l_f1': rouge_scores['rouge-l']['f'],
            'bert_score_f1': F1.mean().item()
        }
    
    def _evaluate_equations(self, original: str, generated: str) -> Dict:
        """Evaluate equation handling accuracy"""
        
        # Extract equations from both texts
        original_eqs = self._extract_equations(original)
        generated_eqs = self._extract_equations(generated)
        
        if not original_eqs:
            return {'equation_count': 0, 'accuracy': 1.0}
        
        # Count correctly preserved equations
        correct_eqs = sum(1 for oeq in original_eqs 
                         if any(oeq in geq for geq in generated_eqs))
        
        equation_accuracy = correct_eqs / len(original_eqs) if original_eqs else 1.0
        
        # Semantic similarity of equations
        eq_similarity = self._calculate_equation_similarity(
            original_eqs, generated_eqs
        )
        
        return {
            'original_equation_count': len(original_eqs),
            'preserved_equation_count': correct_eqs,
            'accuracy': equation_accuracy,
            'semantic_similarity': eq_similarity
        }
    
    def _evaluate_structure(self, original: str, generated: str) -> Dict:
        """Evaluate preservation of paper structure"""
        
        # Identify sections in both
        original_sections = self._identify_sections(original)
        generated_sections = self._identify_sections(generated)
        
        # Calculate section preservation
        section_names_original = set(original_sections.keys())
        section_names_generated = set(generated_sections.keys())
        
        preserved_sections = section_names_original.intersection(section_names_generated)
        
        return {
            'section_preservation_rate': len(preserved_sections) / len(section_names_original) if section_names_original else 1.0,
            'original_sections': list(section_names_original),
            'preserved_sections': list(preserved_sections)
        }
    
    def _evaluate_audio_quality(self, audio_file: str) -> Dict:
        """Evaluate audio quality metrics"""
        
        import librosa
        
        # Load audio
        y, sr = librosa.load(audio_file)
        
        # Calculate audio features
        rms = librosa.feature.rms(y=y).mean()
        zero_crossings = librosa.feature.zero_crossing_rate(y).mean()
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        
        # Estimate SNR (simplified)
        noise_floor = np.percentile(np.abs(y), 10)
        signal_power = np.mean(y**2)
        noise_power = noise_floor**2
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return {
            'rms_energy': float(rms),
            'zero_crossing_rate': float(zero_crossings),
            'spectral_centroid': float(spectral_centroids),
            'estimated_snr': float(snr),
            'duration_seconds': len(y) / sr
        }
    
    def _evaluate_coverage(self, original: str, generated: str) -> Dict:
        """Evaluate content coverage"""
        
        # Calculate n-gram coverage
        original_words = set(original.lower().split())
        generated_words = set(generated.lower().split())
        
        word_coverage = len(generated_words.intersection(original_words)) / len(original_words) if original_words else 1.0
        
        # Key phrase coverage (using simple TF-IDF approach)
        original_phrases = self._extract_key_phrases(original)
        generated_phrases = self._extract_key_phrases(generated)
        
        phrase_coverage = len(original_phrases.intersection(generated_phrases)) / len(original_phrases) if original_phrases else 1.0
        
        return {
            'word_coverage': word_coverage,
            'key_phrase_coverage': phrase_coverage,
            'content_retention_rate': (word_coverage + phrase_coverage) / 2
        }
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Calculate weighted overall score"""
        
        weights = {
            'text_preservation': 0.3,
            'equation_accuracy': 0.25,
            'structure_preservation': 0.2,
            'content_coverage': 0.25
        }
        
        overall = 0.0
        
        # Get text preservation score (average of ROUGE and BERTScore)
        if 'text_preservation' in metrics:
            tp = metrics['text_preservation']
            text_score = (tp.get('rouge_1_f1', 0) + tp.get('bert_score_f1', 0)) / 2
            overall += text_score * weights['text_preservation']
        
        # Equation accuracy
        if 'equation_accuracy' in metrics:
            eq = metrics['equation_accuracy']
            equation_score = eq.get('accuracy', 0) * eq.get('semantic_similarity', 1)
            overall += equation_score * weights['equation_accuracy']
        
        # Structure preservation
        if 'structure_preservation' in metrics:
            sp = metrics['structure_preservation']
            structure_score = sp.get('section_preservation_rate', 0)
            overall += structure_score * weights['structure_preservation']
        
        # Content coverage
        if 'content_coverage' in metrics:
            cc = metrics['content_coverage']
            coverage_score = cc.get('content_retention_rate', 0)
            overall += coverage_score * weights['content_coverage']
        
        return overall
    
    def _extract_equations(self, text: str) -> List[str]:
        """Helper to extract equations from text"""
        import re
        equation_patterns = [
            r'\$\$(.*?)\$\$',
            r'\$(.*?)\$',
            r'\\\[(.*?)\\\]',
            r'\\\((.*?)\\\)'
        ]
        
        equations = []
        for pattern in equation_patterns:
            equations.extend(re.findall(pattern, text, re.DOTALL))
        
        return equations
    
    def _identify_sections(self, text: str) -> Dict:
        """Helper to identify sections in text"""
        # Simple section detection based on common patterns
        import re
        
        section_patterns = [
            r'^#{1,3}\s+(.*?)$',  # Markdown headers
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS lines
            r'^(Abstract|Introduction|Methodology|Results|Discussion|Conclusion|References)$'  # Common sections
        ]
        
        sections = {}
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            for pattern in section_patterns:
                match = re.search(pattern, line.strip())
                if match:
                    section_name = match.group(1).strip()
                    sections[section_name] = i
        
        return sections
    
    def _extract_key_phrases(self, text: str) -> set:
        """Extract key phrases using simple frequency-based approach"""
        from collections import Counter
        import re
        
        # Tokenize and get n-grams
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Get bigrams as simple key phrases
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        
        # Count frequencies
        bigram_counts = Counter(bigrams)
        
        # Return most common bigrams
        return set([phrase for phrase, count in bigram_counts.most_common(20)])
    
    def _calculate_equation_similarity(self, eqs1: List[str], eqs2: List[str]) -> float:
        """Calculate semantic similarity between equation sets"""
        if not eqs1 or not eqs2:
            return 0.0
        
        # Use string similarity as proxy
        similarities = []
        
        for eq1 in eqs1:
            best_sim = 0
            for eq2 in eqs2:
                # Simple Jaccard similarity on characters
                set1, set2 = set(eq1), set(eq2)
                jaccard = len(set1.intersection(set2)) / len(set1.union(set2)) if set1.union(set2) else 0
                best_sim = max(best_sim, jaccard)
            similarities.append(best_sim)
        
        return np.mean(similarities) if similarities else 0.0pip uninstall fitz -ypip uninstall fitz -y