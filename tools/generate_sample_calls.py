"""
Örnek çağrı merkezi ses dosyaları üretici.
====================================================================
İki farklı Türkçe sesle (temsilci + müşteri) sentetik çağrı kayıtları
oluşturur. Test/demo için farklı kalite seviyelerinde örnekler üretir.

Kurulum (bir kez):
    pip install edge-tts pydub
    # ffmpeg zaten kurulu olmalı (brew install ffmpeg)

Çalıştırma:
    python tools/generate_sample_calls.py

Çıktı:
    data/sample_calls/  klasörüne .wav dosyaları yazılır.
    Bu dosyaları doğrudan Streamlit arayüzüne yükleyip test edebilirsin.
"""

import asyncio
import os
import tempfile

import edge_tts
from pydub import AudioSegment

# İki ayrı konuşmacı için iki ayrı Türkçe ses (diarization'ın ayırması için
# erkek/kadın seçildi — ses karakterleri belirgin farklı).
VOICES = {
    "representative": "tr-TR-AhmetNeural",
    "customer": "tr-TR-EmelNeural",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_calls")

# Konuşmacılar arası kısa sessizlik (ms) — gerçek çağrı ritmi için.
PAUSE_MS = 450


# ---------------------------------------------------------------------------
# Örnek diyaloglar: (dosya_adı, [(konuşmacı, metin), ...])
# Farklı kalite seviyeleri — skorlama motorunu test etmek için.
# ---------------------------------------------------------------------------
DIALOGS = {
    # İYİ çağrı: selamlama, kimlik doğrulama, empati, çözüm, kapanış.
    "cagri_iyi": [
        ("representative", "İyi günler, ACME Telekom müşteri hizmetlerine hoş geldiniz, ben Ahmet. Size nasıl yardımcı olabilirim?"),
        ("customer", "Merhaba, faturamda anlamadığım bir ek ücret var."),
        ("representative", "Tabii efendim, hemen kontrol ediyorum. Hesabınızı doğrulamak için müşteri numaranızı alabilir miyim?"),
        ("customer", "Tabii, bir iki üç dört beş altı."),
        ("representative", "Teşekkür ederim. Evet, ek ücretin geçen ayki paket değişikliğinizden kaynaklandığını görüyorum."),
        ("customer", "Bunu bana kimse söylemedi açıkçası, biraz canım sıkıldı."),
        ("representative", "Sizi çok iyi anlıyorum, bu konuda bilgilendirmenin daha net yapılması gerekirdi. Hemen sizin için bir düzeltme talebi oluşturuyorum."),
        ("customer", "Çok teşekkür ederim, çok yardımcı oldunuz."),
        ("representative", "Rica ederim efendim. Sonuç hakkında size SMS ile bilgi vereceğiz. Başka bir konuda yardımcı olabilir miyim?"),
        ("customer", "Hayır, bu kadar. İyi günler."),
        ("representative", "İyi günler dilerim, bizi tercih ettiğiniz için teşekkürler."),
    ],
    # ORTA çağrı: çözüm var ama selamlama eksik, empati zayıf.
    "cagri_orta": [
        ("representative", "Alo, buyurun."),
        ("customer", "Merhaba, internetim iki gündür çok yavaş."),
        ("representative", "Hattınıza bakıyorum. Bir sorun görünüyor, teknik ekibe ileteceğim."),
        ("customer", "Ne zaman düzelir peki?"),
        ("representative", "Yirmi dört saat içinde bakılır."),
        ("customer", "Tamam o zaman bekleyeceğim."),
        ("representative", "Başka bir şey var mı?"),
        ("customer", "Yok, hoşça kalın."),
    ],
    # ZAYIF çağrı: ilgisiz, kaba ton, çözüm yok.
    "cagri_zayif": [
        ("representative", "Efendim ne istiyorsunuz?"),
        ("customer", "Faturam çok yüksek geldi, bir açıklama istiyorum."),
        ("representative", "Ne yapayım yani, kullanmışsınız ödeyeceksiniz."),
        ("customer", "Ama bu kadar kullanmadım, çok sinirlendim şu an."),
        ("representative", "Bakın benim elimden bir şey gelmez, başka kanaldan deneyin."),
        ("customer", "Bu nasıl bir hizmet anlayışı böyle."),
        ("representative", "Kapatıyorum, iyi günler."),
    ],
    # ÖLÜ HAVA (habersiz): temsilci uyarı yapmadan ~13 sn sessiz kalıyor.
    # 3'üncü eleman = o replikten SONRA eklenecek ekstra sessizlik (saniye).
    "cagri_olu_hava_habersiz": [
        ("representative", "İyi günler, ACME Telekom, ben Ahmet, size nasıl yardımcı olabilirim?"),
        ("customer", "Faturamda anlamadığım bir ek ücret var."),
        ("representative", "Tamam.", 13),
        ("representative", "Evet, ek ücreti kaldırıyorum."),
        ("customer", "Teşekkür ederim."),
        ("representative", "Rica ederim, iyi günler dilerim."),
    ],
    # ÖLÜ HAVA (anonslu): temsilci önce "bir saniye, kontrol ediyorum" diyor.
    "cagri_olu_hava_anonslu": [
        ("representative", "İyi günler, ACME Telekom, ben Ahmet, size nasıl yardımcı olabilirim?"),
        ("customer", "Faturamda anlamadığım bir ek ücret var."),
        ("representative", "Tabii, bir saniye, kontrol ediyorum, lütfen hatta kalın.", 13),
        ("representative", "Teşekkürler beklediğiniz için, ek ücreti kaldırdım."),
        ("customer", "Teşekkür ederim."),
        ("representative", "Rica ederim, iyi günler dilerim."),
    ],
}


async def _synthesize_line(text, voice, out_path):
    """Tek bir konuşma satırını edge-tts ile sese çevirir (mp3)."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


async def _build_call(turns):
    """Bir diyaloğu tek bir AudioSegment olarak birleştirir.

    Her tur (speaker, text) ya da (speaker, text, extra_silence_sec) olabilir;
    üçüncü eleman varsa o replikten sonra o kadar saniye sessizlik eklenir
    (ölü hava testleri için).
    """
    pause = AudioSegment.silent(duration=PAUSE_MS)
    call = AudioSegment.silent(duration=300)  # baştan kısa sessizlik

    with tempfile.TemporaryDirectory() as tmp:
        for index, turn in enumerate(turns):
            speaker, text = turn[0], turn[1]
            extra_silence_sec = turn[2] if len(turn) > 2 else 0
            voice = VOICES[speaker]
            line_path = os.path.join(tmp, f"line_{index}.mp3")
            await _synthesize_line(text, voice, line_path)
            segment = AudioSegment.from_file(line_path, format="mp3")
            call += segment + pause
            if extra_silence_sec:
                call += AudioSegment.silent(duration=int(extra_silence_sec * 1000))

    return call


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, turns in DIALOGS.items():
        print(f"Üretiliyor: {name} ...")
        call = await _build_call(turns)
        out_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
        # 16kHz mono — modelin beklediği formata yakın, dosya küçük.
        call = call.set_frame_rate(16000).set_channels(1)
        call.export(out_path, format="wav")
        print(f"  -> {os.path.abspath(out_path)}  ({len(call) / 1000:.1f} sn)")

    print("\nBitti. data/sample_calls/ klasöründeki .wav dosyalarını arayüze yükleyebilirsin.")


if __name__ == "__main__":
    asyncio.run(main())
