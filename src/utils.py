"""
Constants and utilities for Marathi speech processing.
Defines verified token IDs, audio vocabulary layout, and special tokens.
"""

# Vocabulary & Audio Codebook Layout
VOCAB_SIZE = 156960
AUDIO_TOKEN_BASE = 128266
NUM_CODEBOOKS = 7
CODEBOOK_SIZE = 4096
TOTAL_AUDIO_TOKENS = NUM_CODEBOOKS * CODEBOOK_SIZE  # 28672
AUDIO_TOKEN_END = AUDIO_TOKEN_BASE + TOTAL_AUDIO_TOKENS - 1  # 156937

SAMPLE_RATE = 24000
TOKENS_PER_FRAME = 7

# Stock Llama-3 Tokens
BOS_TOKEN_ID = 128000
EOT_TOKEN_ID = 128009

# Project Control Tokens
START_OF_SPEECH_ID = 128257
END_OF_SPEECH_ID = 128258
START_OF_HUMAN_ID = 128259
END_OF_HUMAN_ID = 128260
START_OF_AI_ID = 128261
END_OF_AI_ID = 128262
PAD_TOKEN_ID = 128263

# Delimiter Tokens
SPEAKER_START_ID = 156938  # <|speaker>
SPEAKER_END_ID = 156939    # <speaker|>
STYLE_START_ID = 156940    # <|style>
STYLE_END_ID = 156941      # <style|>
ENV_START_ID = 156942      # <|env>
ENV_END_ID = 156943        # <env|>

# Non-Verbal Tokens
NV_TOKENS = {
    "breath": 156944,
    "stammer": 156945,
    "throat": 156946,
    "laugh": 156947,
    "swallow": 156948,
    "sniff": 156949,
    "sigh": 156950,
    "cough": 156951,
    "hum": 156952,
    "gasp": 156953,
    "wheeze": 156954,
    "sneeze": 156955,
    "snort": 156956,
    "yawn": 156957,
    "groan": 156958,
    "burp": 156959,
}
