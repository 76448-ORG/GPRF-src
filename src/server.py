from TextAnalyser import Analyser as TextEng
from AudioAnalyser import Analyser as AudioEng

@app.post("/api/v1/query")
async def query_gprf(message: str, audio_path: str = None):
    # 1. Feature Extraction
    t_diag = TextEng(message).get_diagnostics()
    a_diag = AudioEng(audio_path).get_analysis() if audio_path else {}

    # 2. Heuristic Mapping to JAST-V (Joy, Anger, Surprise, Trust)
    # Joy: High vocab-diversity + low jitter
    joy = (t_diag['vocabulary-diversity'] * 0.7) + (1 - a_diag.get('idiosyncrasies', {}).get('jitter_local_pct', 0.5))
    
    # Anger: High intensity + high cap-ratio
    anger = (t_diag['capitalization-ratio'] * 0.8) + (a_diag.get('intensity', {}).get('mean_db', 50) / 100)
    
    # Surprise: High f0_std + high punc-freq
    surprise = (a_diag.get('pitch', {}).get('stdev_f0_hz', 0) / 50) + (sum(t_diag['punctuation-frequency'].values()))
    
    # Trust: Low shimmer + low typo-rate
    trust = (1 - t_diag['typos-rate']) * 0.6 + (1 - a_diag.get('idiosyncrasies', {}).get('shimmer_local_pct', 0.5))

    jast_v = [min(1.0, float(x)) for x in [joy, anger, surprise, trust]]
    return {"jast_v": jast_v, "status": "processed"}
