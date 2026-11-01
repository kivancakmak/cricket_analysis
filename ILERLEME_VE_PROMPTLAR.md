# Kriket AI Coach — Portföy Demosu: İlerleme Planı ve Promptlar

**Hedef:** 1–2 günde, başvuruya link olarak koyabileceğin çalışan bir demo.
Video yükle → iskelet takibi → kritik anlar → ölçümler → 100 üzerinden puan + düzeltme önerisi → iskeletli video çıktısı.
Sonunda: **GitHub repo + Hugging Face Spaces canlı link + 30–60 sn ekran kaydı.**

**Teknoloji:** Python · MediaPipe Pose Landmarker (Tasks API) · OpenCV · Gradio (web arayüzü) · YAML kural dosyaları

---

## 0. Başlamadan önce bulman / hazırlaman gerekenler

- [ ] **Test videoları (3–6 adet, 3–6 sn, yandan çekim).**
  - En iyisi: **Kendin veya bir arkadaşın** yandan çekilmiş forward defence / front-foot drive hareketi yapsın. Elinde kriket sopası yoksa düz bir sopa/tahta yeter; demo için sadece vücut hareketi önemli. Böylece telif sorunu olmaz ve "bütçe Android telefonu, dış mekân" koşulunu da gerçekten test etmiş olursun.
  - Ek olarak referans için YouTube'da *"forward defence side on slow motion"*, *"cover drive side view coaching"* aratıp izle. **Bu videoları public repoya/Spaces'e koyma** (telif); sadece kendi denemen için kullan.
  - Çekim ipuçları: telefon bel hizasında, tam yandan, oyuncunun tüm vücudu kadrajda, 30 veya 60 fps.
  - En az 1 tane bilerek **kötü** video çek (yarım kadraj, karanlık, çok uzak) → "please film again" özelliğini test etmek için.
- [ ] **Antrenörlük referansı:** Forward defence ve cover drive için 4–6 temel kontrol noktası (kafa ön dizin üstünde, ön diz bükük, adım uzunluğu, denge, ön dirsek yüksek…). Google'da *"forward defence coaching points"*, *"cover drive technique checklist"*. Bunlar kural dosyasına dönüşecek.
- [ ] **Hesaplar:** GitHub (repo için), Hugging Face (ücretsiz Spaces ile canlı demo linki).
- [ ] **Bilgisayarda:** Python 3.10–3.12, git, ffmpeg.

---

## İlerleme Listesi

### Gün 1 — Çekirdek pipeline
- [ ] 1. Proje iskeleti + `CLAUDE.md` repoya kondu
- [ ] 2. Pose çıkarımı: video → kare kare 33 nokta + görünürlük (JSON/NPY'ye kaydediliyor)
- [ ] 3. Overlay: iskelet çizilmiş çıktı videosu
- [ ] 4. Yön/el tespiti (sağ/sol elli, oyuncu hangi yöne bakıyor → "ön ayak" hangisi)
- [ ] 5. Kritik anlar: stance, front-foot landing, contact (yaklaşık)

### Gün 2 — Puanlama, arayüz, yayın
- [ ] 6. Ölçümler (metrics): kafa–ön diz hizası, ön diz açısı, adım uzunluğu, denge, (drive için) ön dirsek
- [ ] 7. YAML kural dosyası + 100 üzerinden puan + alt puanlar + 1–2 düzeltme önerisi
- [ ] 8. Güven kontrolü → "Please film again" yanıtı
- [ ] 9. Gradio web arayüzü (yükle → sonuç videosu + key frame'ler + puan kartı)
- [ ] 10. Basit etiketleme/dışa aktarım (antrenör puanı gir → CSV)
- [ ] 11. Testler + README (mimari şeması, ekran görüntüleri, "Phase 2: bowling" bölümü)
- [ ] 12. Hugging Face Spaces'e yükleme + ekran kaydı
- [ ] 13. Başvuru cevaplarını yaz (linkler hazır)

---

## Claude Code Promptları

> Sırayla yapıştır. Her adım bitince çalıştırıp kendi videonla dene, sonra bir sonrakine geç. Her promptun sonunda Claude Code'dan `PROGRESS.md`'yi güncellemesini istiyoruz.

### Prompt 1 — Proje kurulumu
```
Read CLAUDE.md first. Set up the project structure described there: create the
folders, a requirements.txt (mediapipe, opencv-python, numpy, pyyaml, gradio,
pytest), a .gitignore (ignore data/raw videos, outputs, models/*.task), and an
empty PROGRESS.md with the 12-step checklist from CLAUDE.md. Add a script
scripts/download_model.py that downloads the MediaPipe pose_landmarker_full.task
model into models/. Initialize git. Don't write pipeline logic yet. Show me how
to create a venv and install everything.
```

### Prompt 2 — Pose çıkarımı
```
Implement src/pose.py using the MediaPipe Tasks API PoseLandmarker in VIDEO
running mode (check the installed mediapipe version and use the current API, not
the legacy mp.solutions if it is deprecated). Function extract_pose(video_path)
returns a PoseSequence dataclass: fps, frame size, and per frame the 33 landmarks
(normalized x, y, z, visibility) plus world landmarks, with None for frames where
no person is detected. Apply light temporal smoothing (e.g. One-Euro or
Savitzky-Golay) as a separate function so raw data is kept. Save/load to .npz.
Add a CLI: python -m src.pose data/sample.mp4 -> prints detection rate and mean
visibility. Add a unit test with a synthetic sequence for the smoothing function.
Update PROGRESS.md.
```

### Prompt 3 — İskelet overlay
```
Implement src/overlay.py: render_overlay(video_path, pose_seq, out_path,
key_frames=None, annotations=None) draws the skeleton on every frame (lines +
joints, low-visibility joints faded), and when key_frames are given, labels those
frames ("STANCE", "FRONT FOOT DOWN", "CONTACT") and holds each for ~0.5 s.
Output must be H.264 mp4 that plays in browsers (use ffmpeg re-encode with
-preset veryfast if OpenCV's writer isn't browser-compatible). Also a function
to export key frames as PNGs. CLI: python -m src.overlay data/sample.mp4.
Update PROGRESS.md.
```

### Prompt 4 — Yön ve ön ayak tespiti
```
Implement src/orientation.py. For a side-on batting clip, determine from the pose
sequence: (1) which direction the batter faces in the image (left/right), (2) the
front leg and back leg, (3) right- or left-handed batter if inferable. Use robust
cues over the first ~1 s (e.g. which ankle moves forward during the stride, nose
vs hip x offset). Return values with a confidence score. Write tests with
synthetic sequences for both facing directions. Update PROGRESS.md.
```

### Prompt 5 — Kritik anlar
```
Implement src/phases.py: detect_key_moments(pose_seq, orientation) returns frame
indices + confidence for:
- STANCE: last stable frame before the front foot starts moving (low ankle/hip
  velocity).
- FRONT_FOOT_LANDING: front ankle stops moving forward and its vertical velocity
  ~0 after the stride (use smoothed velocity, normalize by body height so it works
  at any distance).
- CONTACT (approximate, no bat/ball tracking yet): frame of peak forward wrist
  speed at/after front foot landing, within a window.
Document the heuristics clearly in docstrings and keep thresholds in a config
dict, because coaches' footage will need tuning later. Add a debug plot script
(matplotlib) showing ankle/wrist velocity curves with detected moments marked.
Test on my clips in data/ and show me the plots. Update PROGRESS.md.
```

### Prompt 6 — Ölçümler
```
Implement src/metrics.py. At the key moments compute, all normalized by the
batter's body scale (e.g. hip-to-shoulder length) so they are camera-distance
independent:
- head_over_front_knee: horizontal offset nose/head vs front knee at
  FRONT_FOOT_LANDING and CONTACT (signed, + = head ahead of knee)
- front_knee_angle: hip-knee-ankle angle at landing
- stride_length: front-to-back ankle distance / body height
- balance: horizontal position of hip center relative to the base between the
  feet (0 = over back foot, 1 = over front foot), plus head stability (head
  movement between landing and contact)
- front_elbow_height: for the drive, front elbow vs front wrist/shoulder height
- spine_tilt: shoulder-mid to hip-mid angle from vertical
Each metric returns value + confidence (from landmark visibilities). Unit tests
with hand-built poses where the answer is known. Update PROGRESS.md.
```

### Prompt 7 — Kural dosyası ve puanlama
```
Implement a data-driven scoring engine in src/scoring.py. Rules live in
rules/forward_defence.yaml and rules/front_foot_drive.yaml — coaches must be able
to edit them without touching code. Each rule: id, metric, moment, ideal range
(or min/max), tolerance band with linear falloff, weight, sub-score category
(Head & Balance, Footwork, Body Shape), and plain-English fix + drill text shown
when the rule fails. Output: total score /100, sub-scores, list of passed/failed
rules with measured values, and the top 1–2 fixes ranked by (weight x deficit).
Fill the YAML files with sensible placeholder values based on standard coaching
points and mark them clearly as "PLACEHOLDER – to be set by coaches". Validate
YAML on load with clear errors. Tests for the scoring math. Update PROGRESS.md.
```

### Prompt 8 — Güven kontrolü / "Please film again"
```
Implement src/quality.py: before scoring, check (1) person detected in >= X% of
frames, (2) mean visibility of key joints (head, hips, knees, ankles, wrists)
above threshold, (3) full body in frame (no key joint at the image border),
(4) clip length 2–8 s, (5) key moments detected with enough confidence,
(6) camera roughly side-on (shoulder width small relative to torso height).
If any fail, return a "please film again" result with a specific, friendly
instruction (e.g. "Move the camera back so your feet are visible"). Thresholds in
config. Test with my bad clip. Update PROGRESS.md.
```

### Prompt 9 — Pipeline + Gradio arayüzü
```
Create src/pipeline.py: analyze(video_path, shot_type) runs quality -> pose ->
orientation -> phases -> metrics -> scoring -> overlay and returns a single
AnalysisResult (JSON-serializable) plus output video path and key-frame images.
Then app.py with Gradio: video upload (or webcam record), shot type dropdown
(Forward Defence / Front-Foot Drive), and results: overlay video, key-frame
gallery, big score, sub-score bars, fixes & drills, and an expandable raw metrics
table. Clean, mobile-friendly layout. Show processing time. Make the pipeline
independent of Gradio so it can later sit behind a cloud API or move on-device.
Update PROGRESS.md.
```

### Prompt 10 — Etiketleme / dışa aktarım
```
Add a "Coach labelling" tab in the Gradio app: coach loads a clip, sees the AI's
key frames and score, can adjust key-moment frames with a slider, enter their own
score per rule (pass/fail) and an overall score, and save. Store labels in
labels/labels.csv (clip id, shot type, coach id, per-rule labels, AI values).
Add scripts/agreement_report.py that computes AI-vs-coach agreement: per-rule
accuracy, Cohen's kappa, key-frame timing error in ms, and overall score
correlation/MAE, output as a markdown report. Update PROGRESS.md.
```

### Prompt 11 — Test, temizlik, README
```
Run all tests and fix failures. Add type hints and docstrings where missing, run
ruff. Then write README.md for a client audience: one-line pitch, demo GIF/
screenshot placeholders, how it works (mermaid architecture diagram:
video -> quality gate -> pose -> phases -> metrics -> rule engine -> feedback),
how to run locally, how coaches edit rules YAML, how accuracy is measured, known
limitations (no bat/ball tracking yet, contact is approximate, thresholds are
placeholders), and a "Phase 2: Bowling" section explaining how the modular
design extends (new phase detector + new metrics + new rules file, same engine).
Update PROGRESS.md.
```

### Prompt 12 — Hugging Face Spaces
```
Prepare the repo for deployment as a Hugging Face Space (Gradio SDK): add the
Space README header (sdk, app_file), make the model download happen at startup if
models/ is empty, make sure it runs on CPU-only free tier, and include 1–2 of MY
OWN sample clips as Gradio examples (not YouTube clips). Give me exact steps to
create the Space and push. Update PROGRESS.md.
```

---

## Bittiğinde elinde olması gerekenler
- [ ] GitHub repo linki (temiz README, testler, kural YAML'ları)
- [ ] Hugging Face Spaces canlı demo linki
- [ ] 30–60 sn ekran kaydı (Loom veya YouTube "unlisted"): video yükle → iskelet → puan → düzeltme önerisi → kötü videoda "please film again"
- [ ] Kendi videolarında birkaç sonuç ekran görüntüsü

Bunlar hazır olunca 6 başvuru sorusunun cevaplarını bu linklerle birlikte yazarız.

## Sık karşılaşılabilecek sorunlar
- **MediaPipe API farkı:** Yeni sürümlerde `mp.solutions.pose` yerine `mediapipe.tasks` PoseLandmarker kullanılır. Hata alırsan Claude Code'a sürümü kontrol ettir.
- **Video tarayıcıda oynamıyor:** OpenCV'nin mp4v codec'i tarayıcıda açılmaz → ffmpeg ile H.264'e çevir (Prompt 3'te var).
- **Kritik anlar yanlış kare:** Debug grafiklerine bak (Prompt 5), eşikleri config'den ayarla. Bu normal; gerçek projede antrenör verisiyle ayarlanacak kısım tam olarak bu.
- **Uzak arka ayak / kol görünmüyor:** Yandan çekimde uzak taraf kapanır; ölçümleri kameraya yakın taraf ve ön/arka ayak ayrımıyla kur, düşük görünürlükte güveni düşür.
