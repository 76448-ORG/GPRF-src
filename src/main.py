from .setup import main as Setup
Setup()

import numpy as np
import json
from collections import namedtuple as ntuple
from NEP import AudioAnalyser, TextAnalyser, VideoAnalyser

# --- Core Data Structures ---
EToken = ntuple(
    'EToken', 
    ['logical_schema', 'emotion_schema', 'jast_v', 'current_abstract']
)
ThresholdValues = ntuple(
    'ThresholdValues',
    ['joy', 'anger', 'surprise', 'trust']
)

class GPRF:
    """
    Generative & Programmable Response Framework - Affective Core.
    Orchestrates multi-modal analysis to generate ETokens.
    """
    def __init__(self, settings_path: str = "settings.json"):
        with open(settings_path, 'r') as f:
            self.settings = json.load(f)
        self.dimensions = self.settings['affective_model']['dimensions']
        self.thresholds = ThresholdValues(**self.settings['domain_routing']['classifier_threshold'])

    def _compute_jast_vector(self, t_diag: dict, a_diag: dict) -> list:
        """
        Heuristic Mapping Engine to achieve >50% accuracy on baseline dialogues.
        Maps extracted features to [Joy, Anger, Surprise, Trust].
        """
        # 1. JOY: High vocab diversity, use of slang/emojis, stable pitch.
        joy = (t_diag.get('vocabulary-diversity', 0) * self.thresholds.joy) + \
              (t_diag.get('emoji-rate', 0) * 10) + \
              (1 - a_diag.get('idiosyncrasies', {}).get('jitter_local_pct', 0.5) / 100)
        
        # 2. ANGER: High capitalization, high intensity (dB), faster syllabic rate.
        anger = (t_diag.get('capitalization-ratio', 0) * self.thresholds.anger) + \
                (a_diag.get('intensity', {}).get('mean_db', 40) / 100) + \
                (a_diag.get('rhythm', {}).get('syllabic_rate_proxy', 0) / 10)

        # 3. SURPRISE: High punctuation frequency (?!), high pitch deviation.
        punc_freq = sum(t_diag.get('punctuation-frequency', {}).values())
        surprise = (punc_freq * self.thresholds.surprise) + \
                   (a_diag.get('pitch', {}).get('stdev_f0_hz', 0) / 100)

        # 4. TRUST: Low typo rate, standard quoting, low shimmer (voice stability).
        trust = (1 - t_diag.get('typos-rate', self.thresholds.trust)) + \
                (1 - a_diag.get('idiosyncrasies', {}).get('shimmer_local_pct', 0.5) / 100)

        # Normalize and clip vectors to [0.0, 1.0]
        vector = [min(1.0, float(x)) for x in [joy, anger, surprise, trust]]
        return vector

    def extract_etoken(self, text: str = None, audio_path: str = None) -> EToken:
        """
        Processes inputs and generates the Emotional Token (EToken).
        """
        # 1. Feature Extraction
        t_diag = TextAnalyser(text).get_diagnostics() if text else {}
        a_diag = AudioAnalyser(audio_path).get_analysis() if audio_path else {}

        # 2. Generate JAST-V Vector
        jast_v = self._compute_jast_vector(t_diag, a_diag)

        # 3. Construct Schemas
        # Logical: Cleaned text for rational processing
        logical_schema = text if text else ""
        
        # Emotional: Metric-augmented string for semantic context
        delta_prefix = f"[JAST-V: {json.dumps(dict(zip(self.dimensions, jast_v)))}]"
        emotion_schema = f"{delta_prefix} {text}" if text else delta_prefix

        return EToken(
            logical_schema=logical_schema,
            emotion_schema=emotion_schema,
            jast_v=jast_v,
            current_abstract={"text": t_diag, "audio": a_diag}
        )
