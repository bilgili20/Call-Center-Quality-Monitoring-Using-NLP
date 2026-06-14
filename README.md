# Türkçe Çağrı Merkezi Kalite Skoru Analizi

Bu proje, Türkçe çağrı merkezi görüşmelerinden temsilci performansına yönelik 0-100 arası kalite skoru üretir. Skor, fine-tuned BERTurk regresyon modelinden gelir; uygulamada ek kural tabanlı ceza veya hard-coded kalite skorlama mantığı kullanılmaz.

## Özellikler

- Metin transcript analizi
- Ses dosyası yükleme ile transkripsiyon ve analiz
- WhisperX ile yerel transkripsiyon ve diarization
- Konuşmacıları müşteri / temsilci olarak manuel eşleştirme
- BERTurk regresyon modeli ile 0-100 kalite skoru üretimi

## Model

- Model tipi: BERTurk fine-tuned regression
- Beklenen giriş formatı:

```text
customer: Merhaba, faturam hakkında bilgi almak istiyorum.
representative: Merhaba, memnuniyetle yardımcı olurum.
```

Model eğitiminde etiketler 0-1 aralığında normalize edildiği için inference sırasında ham model çıktısı 100 ile çarpılır ve 0-100 aralığına kırpılır.

Kalite sınıfları yalnızca arayüz ve raporlama içindir:

- Excellent / Mükemmel: score >= 85
- Good / İyi: score >= 70
- Average / Orta: score >= 55
- Poor / Zayıf: score < 55

## Model Dosyaları

Model dosyaları GitHub'a dahil edilmez. Fine-tuned model dosyalarını şu klasöre yerleştirin:

```text
models/berturk_call_quality_regression
```

Beklenen dosyalar örnek olarak:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
training_args.bin
```

Tokenizer önce yerel model klasöründen yüklenir. Yerel tokenizer yüklenemezse uygulama şu fallback tokenizer'ı kullanır:

```text
dbmdz/bert-base-turkish-cased
```

Fine-tuned modelin kendisi her zaman `models/berturk_call_quality_regression` klasöründen yüklenir.

## Kurulum

Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

WhisperX ses işleme için sistemde `ffmpeg` gerektirir. macOS için örnek kurulum:

```bash
brew install ffmpeg
```

Diarization için Hugging Face token gerekebilir. Token'ı ortam değişkeni olarak tanımlayın:

```bash
export HF_TOKEN="your_hugging_face_token"
```

Token verilmezse WhisperX transkripsiyon yapabilir, ancak konuşmacı etiketleri eksik olabilir.

## Uygulamayı Çalıştırma

```bash
streamlit run app.py
```

Uygulamada iki ana sekme bulunur:

- Metin ile Analiz
- Ses Dosyası Yükle

Ses akışlarında WhisperX ham konuşmacı transcriptini üretir. Kullanıcı daha sonra konuşmacıları müşteri veya temsilci olarak eşleştirir ve transcript BERTurk kalite modeline gönderilir.
