/* qwen3tts_c_api.h — C API wrapper for qwen3-tts.cpp (Nim FFI) */
#ifndef QWEN3TTS_C_API_H
#define QWEN3TTS_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Feature macro: this header exposes qwen3_tts_set_pcm_callback(). Hosts that
 * build against a pinned engine can #ifdef on it to stay compatible with pins
 * that predate PCM streaming. */
#define QWEN3_TTS_HAS_PCM_STREAMING 1

/* Opaque handle */
typedef struct Qwen3Tts Qwen3Tts;

/* Generation parameters */
typedef struct Qwen3TtsParams {
    int32_t max_audio_tokens;    /* default: 4096 */
    float   temperature;         /* default: 0.9, 0=greedy */
    float   top_p;               /* default: 1.0 */
    int32_t top_k;               /* default: 50, 0=disabled */
    int32_t n_threads;           /* default: 4 */
    float   repetition_penalty;  /* default: 1.05 */
    int32_t language_id;         /* 2050=en, 2058=ja, 2055=zh, etc. */
} Qwen3TtsParams;

/* Generated audio result */
typedef struct Qwen3TtsAudio {
    const float* samples;  /* PCM float32 mono */
    int32_t n_samples;
    int32_t sample_rate;   /* always 24000 */
} Qwen3TtsAudio;

/* Fill params with defaults */
void qwen3_tts_default_params(Qwen3TtsParams* params);

/* Create TTS engine and load models from directory.
 * model_dir must contain a tier GGUF (qwen3-tts-0.6b-f16.gguf and/or
 * qwen3-tts-1.7b-f16.gguf; select with QWEN3_TTS_TIER / QWEN3_TTS_MODEL) and
 * qwen3-tts-tokenizer-f16.gguf.
 * Returns NULL on failure. */
Qwen3Tts* qwen3_tts_create(const char* model_dir, int32_t n_threads);

/* Check if models are loaded */
int qwen3_tts_is_loaded(const Qwen3Tts* tts);

/* Synthesize text to audio. Returns NULL on failure.
 * Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize(
    Qwen3Tts* tts,
    const char* text,
    const Qwen3TtsParams* params);

/* Get sample rate (always 24000) */
int32_t qwen3_tts_sample_rate(const Qwen3Tts* tts);

/* Free generated audio */
void qwen3_tts_free_audio(Qwen3TtsAudio* audio);

/* Destroy TTS engine */
void qwen3_tts_destroy(Qwen3Tts* tts);

/* Synthesize with voice cloning from WAV file.
 * reference_audio_path: path to reference WAV (24kHz mono recommended).
 * Returns NULL on failure. Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize_with_voice_file(
    Qwen3Tts* tts,
    const char* text,
    const char* reference_audio_path,
    const Qwen3TtsParams* params);

/* Synthesize with voice cloning from raw samples.
 * ref_samples: 24kHz mono float32 normalized to [-1, 1].
 * Returns NULL on failure. Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize_with_voice_samples(
    Qwen3Tts* tts,
    const char* text,
    const float* ref_samples,
    int32_t n_ref_samples,
    const Qwen3TtsParams* params);

/* Extract speaker embedding from WAV file (for caching).
 * embedding_out: caller-allocated buffer for the embedding.
 * max_size: size of embedding_out in floats.
 * Returns the actual embedding size (typically 1024), or -1 on failure. */
int32_t qwen3_tts_extract_embedding_file(
    Qwen3Tts* tts,
    const char* reference_audio_path,
    float* embedding_out,
    int32_t max_size);

/* Number of floats a speaker embedding must have for the loaded model
 * (talker hidden_size: 1024 on 0.6B, 2048 on 1.7B). 0 before load. */
int32_t qwen3_tts_speaker_embedding_size(const Qwen3Tts* tts);

/* Synthesize with pre-computed speaker embedding (skips encoder).
 * embedding: speaker embedding from qwen3_tts_extract_embedding_file().
 * embedding_size: must match qwen3_tts_speaker_embedding_size().
 * Returns NULL on failure. Caller must free with qwen3_tts_free_audio(). */
Qwen3TtsAudio* qwen3_tts_synthesize_with_embedding(
    Qwen3Tts* tts,
    const char* text,
    const float* embedding,
    int32_t embedding_size,
    const Qwen3TtsParams* params);

/* PCM streaming callback. Invoked with one decoded chunk while synthesis is
 * still running, on the thread that called qwen3_tts_synthesize*.
 * samples: mono float32 at sample_rate, valid for the duration of the call only.
 * Return 0 to abort synthesis (the synthesize call then returns NULL). */
typedef int (*Qwen3TtsPcmCallback)(
    const float* samples,
    int32_t n_samples,
    int32_t sample_rate,
    void* user_data);

/* Enable PCM streaming for subsequent qwen3_tts_synthesize* calls: code
 * generation and vocoder decode interleave on the calling thread, and cb is
 * invoked as soon as each chunk's codes exist. The synthesize call still
 * returns the complete audio, so callers that ignore the callback see no
 * behaviour change.
 *
 * cb = NULL restores whole-utterance synthesis, which is the default.
 * chunk_frames <= 0 uses the engine default (QWEN3_TTS_STREAM_CHUNK_FRAMES,
 * 12 codec frames ~= 0.96 s of audio).
 *
 * The streamed chunks concatenate to the same waveform the whole-utterance path
 * produces for the same chunk schedule; they are not sample-identical to a
 * single-graph decode of the whole utterance, which has unbounded left context. */
void qwen3_tts_set_pcm_callback(
    Qwen3Tts* tts,
    Qwen3TtsPcmCallback cb,
    void* user_data,
    int32_t chunk_frames);

/* Get last error message (or empty string) */
const char* qwen3_tts_get_error(const Qwen3Tts* tts);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3TTS_C_API_H */
