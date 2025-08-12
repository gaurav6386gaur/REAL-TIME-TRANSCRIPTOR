from flask import Flask, render_template, request, send_file
import os
import whisper
from docx import Document
from datetime import datetime
from werkzeug.utils import secure_filename
import warnings
import librosa
import numpy as np
import ffmpeg
from scipy import signal
import soundfile as sf
import re

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRANSCRIPT_FOLDER'] = 'transcripts'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRANSCRIPT_FOLDER'], exist_ok=True)

print("Loading Whisper model...")
# Changed to 'small' for better language handling, 'medium' or 'large' can be even better if resources allow.
# If you consistently need only English and Hinglish, 'medium.en' might be a good choice.
model = whisper.load_model("small")
print("Model loaded successfully!")

def enhance_audio_quality(input_path):
    """Enhanced audio preprocessing for better clarity"""
    output_path = input_path.replace(".wav", "_enhanced.wav")
    try:
        # Load audio with better parameters for voice
        y, sr = librosa.load(input_path, sr=22050)  # Higher sample rate for better quality

        # Advanced noise reduction
        S = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(S)
        phase = np.angle(S)

        # Better noise estimation using longer initial segment
        noise_duration = min(1.0, len(y) / sr * 0.1)  # 10% or 1 second
        noise_frames = int(noise_duration * sr / 512)
        noise_sample = magnitude[:, :noise_frames]
        noise_floor = np.mean(noise_sample, axis=1, keepdims=True)
        noise_std = np.std(noise_sample, axis=1, keepdims=True)

        # Adaptive spectral gating
        threshold = noise_floor + 2 * noise_std
        mask = magnitude > threshold

        # Smooth the mask to avoid artifacts
        from scipy.ndimage import uniform_filter1d
        mask = uniform_filter1d(mask.astype(float), size=3, axis=1) > 0.5

        # Apply mask with fade edges
        S_clean = S * mask

        # Reconstruct audio
        y_clean = librosa.istft(S_clean, hop_length=512)

        # Voice enhancement: boost mid frequencies (human voice range)
        # Apply bandpass filter for voice range (80Hz - 8kHz)
        from scipy.signal import butter, filtfilt
        nyquist = sr // 2
        low = 80 / nyquist
        high = 8000 / nyquist
        b, a = butter(4, [low, high], btype='band')
        y_filtered = filtfilt(b, a, y_clean)

        # Normalize and apply gentle compression
        y_filtered = librosa.util.normalize(y_filtered)
        y_compressed = np.tanh(y_filtered * 2.0) * 0.8

        # Apply de-emphasis to brighten voice
        y_final = librosa.effects.preemphasis(y_compressed)

        # Save enhanced audio
        sf.write(output_path, y_final, sr)
        return output_path

    except Exception as e:
        print(f"Audio enhancement error: {e}")
        return input_path

def convert_to_mono(input_path):
    """Convert stereo to mono with voice optimization"""
    output_path = input_path.replace(".wav", "_mono.wav")
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path,
                    ac=1,  # mono
                    ar=22050,  # sample rate for better quality
                    af='highpass=f=85,lowpass=f=8000,volume=1.5')  # Voice optimization
            .run(overwrite_output=True, quiet=True)
        )
        return output_path
    except Exception as e:
        print("Mono conversion error:", e)
        return input_path

def detect_ai_voice(audio_path, transcript_text=""):
    """Enhanced AI voice detection with multiple indicators"""
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=30)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        if len(y_trimmed) < sr:
            return False, 0.0

        # Feature extraction
        f0 = librosa.yin(y_trimmed, fmin=50, fmax=400, sr=sr)
        valid_f0 = f0[f0 > 0]

        if len(valid_f0) < 10:
            return True, 0.9  # Very likely AI if no valid pitch

        # AI indicators
        ai_score = 0
        total_checks = 8

        # 1. Pitch stability (AI voices are too stable)
        f0_variance = np.std(valid_f0)
        if f0_variance < 12:  # Too stable
            ai_score += 1

        # 2. Pitch range (AI voices have limited range)
        f0_range = np.ptp(valid_f0)
        if f0_range < 25:  # Too narrow
            ai_score += 1

        # 3. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
        centroid_variance = np.std(spectral_centroids)
        if centroid_variance < 200:  # Too consistent
            ai_score += 1

        # 4. Zero crossing rate consistency
        zcr = librosa.feature.zero_crossing_rate(y_trimmed)[0]
        zcr_variance = np.std(zcr)
        if zcr_variance < 0.005:  # Too consistent
            ai_score += 1

        # 5. Spectral rolloff consistency
        rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)[0]
        rolloff_variance = np.std(rolloff)
        if rolloff_variance < 500:  # Too consistent
            ai_score += 1

        # 6. Unnatural pauses detection
        rms = librosa.feature.rms(y=y_trimmed)[0]
        silence_threshold = np.mean(rms) * 0.1
        silent_frames = rms < silence_threshold

        # Find pause durations
        pause_changes = np.diff(silent_frames.astype(int))
        pause_starts = np.where(pause_changes == 1)[0]
        pause_ends = np.where(pause_changes == -1)[0]

        if len(pause_starts) > 0 and len(pause_ends) > 0:
            # Calculate pause durations
            min_len = min(len(pause_starts), len(pause_ends))
            pause_durations = pause_ends[:min_len] - pause_starts[:min_len]

            # AI tends to have very regular pause patterns
            if len(pause_durations) > 3:
                pause_regularity = np.std(pause_durations)
                if pause_regularity < 2:  # Too regular
                    ai_score += 1

        # 7. Text-based AI detection
        if transcript_text:
            # AI speech patterns
            ai_phrases = [
                'aapko lagayega', 'available hai', 'square feet carpet area',
                'possession project', 'inquiry', 'details chahiye',
                'contact kar sakte hain', 'aapko dikhayenge', 'agar aapko'
            ]

            text_lower = transcript_text.lower()
            ai_phrase_count = sum(1 for phrase in ai_phrases if phrase in text_lower)
            if ai_phrase_count >= 2:
                ai_score += 1

        # 8. Formant analysis (AI voices have unnatural formants)
        # Extract formants using LPC
        try:
            # Simple formant estimation
            frame_length = 2048
            frames = librosa.util.frame(y_trimmed, frame_length=frame_length, hop_length=frame_length//2)

            formant_consistency = []
            for frame in frames.T:
                if np.sum(frame**2) > 0.001:  # Skip silent frames
                    # Auto-correlation based pitch
                    autocorr = np.correlate(frame, frame, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]

                    if len(autocorr) > 100:
                        formant_consistency.append(np.std(autocorr[:100]))

            if formant_consistency:
                avg_formant_variance = np.mean(formant_consistency)
                if avg_formant_variance < 0.01:  # Too consistent formants
                    ai_score += 1
        except:
            pass  # Skip formant analysis if it fails

        confidence = ai_score / total_checks
        is_ai = confidence > 0.4  # Threshold for AI detection

        return is_ai, confidence

    except Exception as e:
        print(f"AI detection error: {e}")
        return False, 0.0

def advanced_gender_detection(audio_path):
    """Improved gender detection with AI voice consideration"""
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=30)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        if len(y_trimmed) < sr:
            return "Unknown", 0.0

        # Fundamental frequency analysis
        f0 = librosa.yin(y_trimmed, fmin=50, fmax=400, sr=sr)
        valid_f0 = f0[f0 > 0]

        if len(valid_f0) < 10:
            return "Unknown", 0.0

        median_f0 = np.median(valid_f0)

        # Additional features
        spectral_centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
        avg_centroid = np.mean(spectral_centroids)

        # Gender scoring
        male_score = 0
        female_score = 0

        # Primary F0 classification
        if median_f0 < 130:  # Typical male range
            male_score += 4
        elif median_f0 < 165:  # Border range
            male_score += 2
            female_score += 1
        elif median_f0 < 220:  # Typical female range
            female_score += 3
        else:  # High female range
            female_score += 4

        # Spectral centroid (voice brightness)
        if avg_centroid < 2200:  # Darker voice
            male_score += 2
        elif avg_centroid > 2800:  # Brighter voice
            female_score += 2

        # Formant estimation (rough)
        if median_f0 < 150 and avg_centroid < 2000:  # Deep male voice
            male_score += 2
        elif median_f0 > 180 and avg_centroid > 2500:  # High female voice
            female_score += 2

        total_score = male_score + female_score
        if total_score == 0:
            return "Unknown", 0.0

        if male_score > female_score:
            confidence = male_score / total_score
            return "Male", confidence
        else:
            confidence = female_score / total_score
            return "Female", confidence

    except Exception as e:
        print(f"Gender detection error: {e}")
        return "Unknown", 0.0

def detect_hindi_content(text):
    """Better Hindi content detection, now prioritizing Hinglish keywords."""
    if not text:
        return False, 0.0

    # Devanagari script detection (still useful for pure Hindi parts)
    devanagari_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars_alpha = len([c for c in text if c.isalpha()])

    devanagari_ratio = devanagari_chars / total_chars_alpha if total_chars_alpha > 0 else 0.0

    # Hindi words in English script (Hinglish) - expanded list
    hinglish_words = [
        'aap', 'hai', 'hain', 'kar', 'kya', 'nahi', 'nahin', 'mein', 'me', 'se', 'ko', 'ka', 'ki', 'ke',
        'bhk', 'flat', 'ghar', 'paisa', 'rupaye', 'lakh', 'crore', 'chahiye', 'available',
        'lagega', 'milega', 'dekh', 'dekhiye', 'samajh', 'samjha', 'baat', 'kaam', 'time',
        'ache', 'accha', 'theek', 'thik', 'bas', 'abhi', 'phir', 'wahan', 'yahan', 'kahan',
        'kaise', 'kyun', 'kyu', 'jab', 'tab', 'sab', 'kuch', 'koi', 'sabko', 'sabse',
        'mere', 'mera', 'meri', 'tumhara', 'tumhari', 'uska', 'uski', 'unka', 'unki',
        'bhi', 'hum', 'hamara', 'apne', 'bataiye', 'pata', 'hai na', 'sirf', 'thoda', 'bahut',
        'liye', 'log', 'logon', 'pehle', 'baad', 'andar', 'bahar', 'upar', 'neeche',
        'aur', 'ya', 'lekin', 'lekin', 'kyunki', 'jiski', 'jiska', 'jismein'
    ]

    text_lower = text.lower()
    # Use word boundaries for more accurate matching
    hinglish_word_count = sum(1 for word in hinglish_words if re.search(r'\b' + re.escape(word) + r'\b', text_lower))
    total_words = len(text_lower.split())

    hinglish_ratio = hinglish_word_count / total_words if total_words > 0 else 0.0

    # Combined score
    # Prioritize Hinglish ratio if Devanagari is low, combine otherwise
    hindi_score = 0.0
    if devanagari_ratio > 0.05: # If there's some Devanagari, it's definitely Hindi/Hinglish
        hindi_score = max(devanagari_ratio, hinglish_ratio)
    else: # Rely more on hinglish keywords if no Devanagari script
        hindi_score = hinglish_ratio

    # Additional patterns focusing on mixed language phrases
    mixed_language_phrases = [
        'hindi me', 'hindi mein', 'baat kar', 'kya chahiye', 'kitna hai',
        'property dekhiye', 'sirf itna', 'mujhe lagta hai', 'aapko kaisa laga'
    ]

    phrase_matches = sum(1 for phrase in mixed_language_phrases if phrase in text_lower)

    # Boost score for phrase matches
    if phrase_matches > 0:
        hindi_score += 0.2 * min(phrase_matches, 3) # Max boost 0.6 for multiple matches

    is_hindi = hindi_score > 0.15 # Lower threshold as we are looking for mixture
    confidence = min(1.0, hindi_score + (phrase_matches * 0.05)) # Slight boost for phrases

    return is_hindi, confidence


def intelligent_speaker_detection(text, primary_gender="Unknown", confidence=0.0, is_ai_detected=False, ai_confidence=0.0):
    """Enhanced speaker detection with AI and Hindi awareness"""
    if not text.strip():
        return "No transcript available."

    # Detect Hindi content for contextual awareness
    has_hindi, hindi_confidence = detect_hindi_content(text)

    # Clean and prepare text
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    # Context detection
    business_keywords = ['ajmera', 'group', 'inquiry', 'property', 'calling', 'interested',
                         'bhk', 'flat', 'possession', 'project', 'square feet', 'carpet area',
                         'site visit', 'brochure', 'price', 'budget']
    greeting_patterns = [r'\b(hi|hello|good morning|good afternoon|good evening|namaste|namaskar)\b']

    is_business_call = any(keyword in text.lower() for keyword in business_keywords)

    # Better conversation segmentation using a wider range of markers
    conversation_markers = [
        r'\b(hi|hello|yes|no|okay|actually|so|but|and|well|aap|haan|acha|nahi|theek|samajh|dekho|boliye)\b',
        r'(\?|\!)\s*',  # Questions and exclamations
        r'\b(sir|madam|ji)\b',
        r'\b(aap|aapko|aapka|tumhe|tumhara|mujhe|mera|hum)\b', # Hindi pronouns
        r'\n\n', # Double newline can indicate a speaker change
        r'(\.\s*){2,}', # Multiple periods might indicate a long pause followed by a new speaker
    ]

    # Split text into segments
    segments = [text]
    for pattern in conversation_markers:
        new_segments = []
        for segment in segments:
            # Use re.split to split by marker and keep the marker if it's significant
            parts = re.split(f'({pattern})', segment, flags=re.IGNORECASE | re.UNICODE)
            for part in parts:
                if part and part.strip():
                    new_segments.append(part.strip())
        segments = new_segments

    # Merge very short segments that are not clear interjections
    merged_segments = []
    current_segment = ""

    for segment in segments:
        # Define what constitutes a "short interjection" that should stand alone
        is_interjection = any(re.match(r'^(hi|hello|yes|no|okay|haan|acha|theek|ji|\?|\!)$', segment.lower()) for segment in segment.split())

        if len(segment.split()) < 4 and current_segment and not is_interjection: # Merge short non-interjections
            current_segment += " " + segment
        else:
            if current_segment:
                merged_segments.append(current_segment)
            current_segment = segment

    if current_segment:
        merged_segments.append(current_segment)

    # Determine speaker labels with AI consideration
    if is_business_call:
        if is_ai_detected and ai_confidence > 0.4:
            if primary_gender == "Female":
                speakers = ["Female AI Agent", "Customer"] # Customer gender is less certain without separate detection
            else:
                speakers = ["AI Agent", "Customer"]
        else:
            if confidence > 0.6: # High confidence in gender for Agent
                if primary_gender == "Male":
                    speakers = ["Male Agent", "Customer"] # Customer gender unknown
                else: # Female
                    speakers = ["Female Agent", "Customer"]
            else:
                speakers = ["Agent", "Customer"] # General labels if gender not confident or AI not detected
    else:
        # General conversation
        if primary_gender == "Male" and confidence > 0.5:
            speakers = ["Male Speaker", "Female Speaker"]
        elif primary_gender == "Female" and confidence > 0.5:
            speakers = ["Female Speaker", "Male Speaker"]
        else:
            speakers = ["Speaker 1", "Speaker 2"]

    # Smart speaker assignment based on content and turn-taking
    formatted_conversation = []
    current_speaker_index = 0 # Assume Agent/Speaker 1 starts by default, can be adjusted

    # Heuristic to determine who starts the conversation if it's a business call
    if is_business_call and merged_segments:
        first_segment_lower = merged_segments[0].lower()
        agent_starters = ['hi', 'hello', 'good', 'ajmera', 'group', 'nisha', 'calling', 'i am calling', 'main baat kar raha hoon']
        customer_starters = ['yes', 'haan', 'ji', 'ok', 'interested', 'mujhe janna hai']

        if any(starter in first_segment_lower for starter in customer_starters) and \
           not any(starter in first_segment_lower for starter in agent_starters):
            current_speaker_index = 1 # Customer starts

    for i, segment in enumerate(merged_segments):
        segment_lower = segment.lower()

        # Simple turn-taking: assume speaker changes if the next segment isn't a continuation of the previous thought
        # This is a basic heuristic and might need more advanced NLP for complex dialogues
        if i > 0:
            prev_segment_lower = merged_segments[i-1].lower()
            # If current segment is a short affirmation/negation after a question
            if ('?' in prev_segment_lower or any(q in prev_segment_lower for q in ['kya', 'how', 'what'])) and \
               any(ans in segment_lower for ans in ['yes', 'no', 'haan', 'nahin', 'acha', 'ok', 'theek']):
                current_speaker_index = 1 - current_speaker_index # Switch speaker

            # If current segment looks like an independent thought or a new question
            elif len(segment.split()) > 5 and not segment_lower.startswith(tuple(['and', 'but', 'so', 'aur', 'lekin', 'to'])):
                if not (segment_lower.startswith(speakers[current_speaker_index].lower().replace(" agent", "").replace(" speaker", ""))): # Avoid switching if it's clearly the same person
                    pass # Don't automatically switch just for length, rely on other markers

            # Explicit speaker change indicators
            if any(marker in segment_lower for marker in ['aap', 'tumhe', 'apko', 'tell me', 'bataiye']) and \
               not any(marker in prev_segment_lower for marker in ['aap', 'tumhe', 'apko']): # If the other person is addressed
                current_speaker_index = 1 - current_speaker_index

        formatted_conversation.append(f"{speakers[current_speaker_index]}: {segment}")
        # Simplistic alternating speaker model for the next turn, unless overridden
        current_speaker_index = 1 - current_speaker_index


    # Add metadata
    metadata = []
    if has_hindi:
        metadata.append(f"Hindi/Hinglish Content Detected: {hindi_confidence:.1%}")
    if is_ai_detected:
        metadata.append(f"AI Voice Detected: {ai_confidence:.1%}")

    result = "\n\n".join(formatted_conversation)
    if metadata:
        result = "=== Analysis ===\n" + " | ".join(metadata) + "\n\n" + result

    return result

def save_transcript_as_txt(content, filename):
    filepath = os.path.join(app.config['TRANSCRIPT_FOLDER'], filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath

def save_transcript_as_docx(content, filename):
    doc = Document()
    doc.add_heading('Enhanced Conversation Transcript', 0)
    doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph()

    lines = content.split('\n')
    for line in lines:
        if line.strip():
            if line.startswith('=== Analysis ==='):
                doc.add_heading('Analysis', level=1)
            elif ':' in line and not line.startswith(' ') and not line.startswith('==='):
                # Speaker line
                try:
                    speaker, text = line.split(':', 1)
                    p = doc.add_paragraph()
                    p.add_run(speaker + ':').bold = True
                    p.add_run(text)
                except ValueError:
                    doc.add_paragraph(line)
            else:
                doc.add_paragraph(line)

    filepath = os.path.join(app.config['TRANSCRIPT_FOLDER'], filename)
    doc.save(filepath)
    return filepath

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return render_template('index.html', error="No file selected")

        file = request.files['audio']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return render_template('index.html', error="Only .mp3, .wav, .m4a, .flac, .ogg files supported")

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            print(f"Processing file: {filename}")

            # Step 1: Convert to mono with voice optimization
            print("Converting to mono...")
            mono_path = convert_to_mono(filepath)

            # Step 2: Enhanced audio quality
            print("Enhancing audio quality...")
            enhanced_path = enhance_audio_quality(mono_path)

            # Step 3: Transcription with better parameters for Hinglish
            print("Starting transcription...")
            start_time = datetime.now()

            # Key change: updated initial_prompt for Hinglish emphasis and language
            result = model.transcribe(
                enhanced_path,
                language='en',  # Explicitly set to English to guide towards English script
                fp16=False,
                verbose=False,  # Enable for debugging
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.7,  # Increased temperature for more flexible output
                no_speech_threshold=0.5,
                logprob_threshold=-1.0,
                initial_prompt="This audio contains conversation in English and Hindi. Please transcribe everything accurately, including Hindi words written using English alphabet (Hinglish)."
            )

            processing_time = (datetime.now() - start_time).total_seconds()
            raw_text = result['text'].strip()
            detected_language = result.get('language', 'mixed') # Whisper's detected lang might still be 'en' but content will be mixed

            print(f"Transcription completed in {processing_time:.1f} seconds")
            print(f"Whisper detected language: {detected_language}")
            print(f"Raw text: {raw_text[:200]}...")

            # Step 4: AI voice detection
            print("Detecting AI voice...")
            is_ai, ai_confidence = detect_ai_voice(enhanced_path, raw_text)

            # Step 5: Gender detection
            print("Detecting speaker characteristics...")
            detected_gender, gender_confidence = advanced_gender_detection(enhanced_path)

            print(f"AI Detection: {is_ai} (confidence: {ai_confidence:.2f})")
            print(f"Gender: {detected_gender} (confidence: {gender_confidence:.2f})")

            # Step 6: Intelligent speaker formatting
            formatted_transcript = intelligent_speaker_detection(
                raw_text, detected_gender, gender_confidence, is_ai, ai_confidence
            )

            # Step 7: Save files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_filename = f"transcript_{timestamp}.txt"
            docx_filename = f"transcript_{timestamp}.docx"

            txt_path = save_transcript_as_txt(formatted_transcript, txt_filename)
            docx_path = save_transcript_as_docx(formatted_transcript, docx_filename)

            # Cleanup temporary files
            for temp_file in [filepath, mono_path, enhanced_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

            print(f"Files saved: {txt_filename}, {docx_filename}")

            # Enhanced results for display
            voice_analysis = []
            if is_ai:
                voice_analysis.append(f"AI Voice ({ai_confidence:.1%} confidence)")
            voice_analysis.append(f"{detected_gender} ({gender_confidence:.1%} confidence)")

            has_hindi, hindi_conf = detect_hindi_content(raw_text)
            if has_hindi:
                voice_analysis.append(f"Hindi/Hinglish Content ({hindi_conf:.1%})")

            return render_template('index.html',
                                   success=True,
                                   txt_file=txt_filename,
                                   docx_file=docx_filename,
                                   original_filename=filename,
                                   processing_time=processing_time,
                                   detected_language=detected_language, # This will be 'en'
                                   voice_analysis=" | ".join(voice_analysis),
                                   transcript_preview=formatted_transcript[:600] + "..." if len(formatted_transcript) > 600 else formatted_transcript)

        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()

            # Cleanup on error
            for temp_file in [filepath,
                              mono_path if 'mono_path' in locals() else None,
                              enhanced_path if 'enhanced_path' in locals() else None]:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
            return render_template('index.html', error=f"Error processing audio: {str(e)}")

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join(app.config['TRANSCRIPT_FOLDER'], filename),
            as_attachment=True
        )
    except FileNotFoundError:
        return "File not found", 404

if __name__ == '__main__':
    
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)