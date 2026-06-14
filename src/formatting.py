SPEAKER_UNKNOWN = "SPEAKER_UNKNOWN"

# ---------------------------------------------------------------------------
# Otomatik temsilci/müşteri ayrımı için AĞIRLIKLI SİNYALLER.
# Her konuşmacı için bir "temsilcilik skoru" hesaplanır:
#   skor = (temsilci sinyalleri) − (müşteri sinyalleri) + (açılış bonusu)
# En yüksek skorlu konuşmacı temsilci kabul edilir. Tek selamlama sinyaline
# göre çok daha dayanıklıdır.
# Anahtar kelimeler "fold" edilmiş (Türkçe karakter sadeleştirilmiş) yazılır.
# ---------------------------------------------------------------------------

# Temsilciye işaret eden sinyaller: (ağırlık, [kalıplar])
AGENT_SIGNALS = [
    (3.0, [  # selamlama / açılış kalıpları
        "hos geldiniz", "hosgeldiniz", "iyi gunler dilerim", "iyi aksamlar dilerim",
        "hattimiza", "bizi tercih",
    ]),
    (3.0, [  # kimlik / doğrulama soruları
        "musteri numara", "musteri no", "dogrulamak icin", "dogrulama",
        "kimlik numara", "tc kimlik", "numaranizi alabilir", "hesabinizi kontrol",
    ]),
    (2.0, [  # firma / kendini tanıtma
        "telekom", "cagri merkezi", "musteri hizmetleri", "ben ", "adim ", "firma",
    ]),
    (2.0, [  # yardım teklifi
        "nasil yardimci", "size nasil", "yardimci olabilirim", "buyurun",
    ]),
    (2.0, [  # somut çözüm / aksiyon
        "talep olustur", "olusturuyorum", "kontrol ediyorum", "kontrol sagliyorum",
        "iletiyorum", "iletecegim", "duzelt", "yonlendiriyorum", "kaydinizi aldim",
        "tanimladim", "guncelliyorum", "aktif ettim",
    ]),
    (1.0, [  # nezaket (temsilci dili)
        "rica ederim", "memnuniyetle", "elbette",
    ]),
]

# Müşteriye işaret eden sinyaller — temsilcilik skorunu düşürür.
CUSTOMER_SIGNALS = [
    (2.0, [  # problem / talep anlatımı
        "faturam", "sorun", "sikayet", "calismiyor", "yavas", "ek ucret",
        "iptal etmek", "param", "hata aliyorum", "anlamadigim", "geri odeme",
        "yuksek geldi", "baglanamiyorum", "kesildi",
    ]),
    (1.0, [  # olumsuz duygu
        "sinirlendim", "kizgin", "biktim", "rezalet", "magdur",
    ]),
]

OPENING_BONUS = 2.0  # çağrıyı ilk açan konuşmacıya temsilcilik avantajı

# Geriye dönük uyumluluk için eski sabit korunuyor (artık AGENT_SIGNALS kullanılır).
GREETING_HINTS = (
    "hos geldiniz", "yardimci olabilirim", "nasil yardimci",
    "buyurun", "iyi gunler dilerim", "cagri merkezi",
)


def _fold(text):
    """Türkçe karakterleri sadeleştirip küçük harfe çevirir."""
    if not text:
        return ""
    text = text.replace("İ", "i").replace("I", "i").replace("ı", "i")
    text = text.lower()
    for a, b in (("ç", "c"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("â", "a")):
        text = text.replace(a, b)
    return text


def _speaker_label(segment):
    return segment.get("speaker") or SPEAKER_UNKNOWN


def get_unique_speakers(segments):
    """Return speaker labels in first-seen order."""
    speakers = []
    for segment in segments or []:
        speaker = _speaker_label(segment)
        if speaker not in speakers:
            speakers.append(speaker)
    return speakers or [SPEAKER_UNKNOWN]


def segments_to_speaker_text(segments):
    """Convert WhisperX segments to readable speaker-labeled text."""
    lines = []
    for segment in segments or []:
        speaker = _speaker_label(segment)
        text = (segment.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _agent_score_for_text(text):
    """Bir konuşmacının tüm metni için temsilcilik skoru (ağırlıklı sinyaller)."""
    folded = _fold(text)
    score = 0.0
    for weight, patterns in AGENT_SIGNALS:
        if any(p in folded for p in patterns):
            score += weight
    for weight, patterns in CUSTOMER_SIGNALS:
        if any(p in folded for p in patterns):
            score -= weight
    # Soru sorma temsilciye işaret eder (en fazla 1.0 katkı).
    score += min(folded.count("?") * 0.3, 1.0)
    return score


def auto_assign_roles(segments):
    """SPEAKER_xx etiketlerini otomatik olarak representative/customer'a eşler.

    Çoklu ağırlıklı sinyal kullanır (tek selamlama kuralından çok daha sağlam):
      - Selamlama / açılış kalıpları
      - Kimlik / müşteri doğrulama soruları
      - Firma adı ve kendini tanıtma
      - Yardım teklifi, somut çözüm/aksiyon ifadeleri, nezaket dili, soru sorma
      - Müşteri sinyalleri (problem anlatımı, olumsuz duygu) skoru DÜŞÜRÜR
      - Çağrıyı ilk açan konuşmacıya küçük bir açılış bonusu
    En yüksek temsilcilik skorlu konuşmacı temsilci, diğerleri müşteri olur.

    Tek konuşmacı (diarization yok / SPEAKER_UNKNOWN) durumunda o konuşmacı
    customer döner; böylece güvenli bir varsayılan sağlanır.
    """
    speakers = get_unique_speakers(segments)

    if len(speakers) == 1:
        return {speakers[0]: "customer"}

    # Her konuşmacının tüm repliklerini birleştir.
    texts = {speaker: [] for speaker in speakers}
    for segment in segments or []:
        speaker = _speaker_label(segment)
        text = (segment.get("text") or "").strip()
        if text:
            texts.setdefault(speaker, []).append(text)

    scores = {
        speaker: _agent_score_for_text(" ".join(lines))
        for speaker, lines in texts.items()
    }

    # Çağrıyı ilk açan konuşmacıya açılış bonusu.
    if speakers:
        scores[speakers[0]] = scores.get(speakers[0], 0.0) + OPENING_BONUS

    # En yüksek skorlu temsilci; tüm skorlar eşitse ilk konuşan temsilci.
    best = max(scores.values()) if scores else 0.0
    if all(v == best for v in scores.values()):
        agent = speakers[0]
    else:
        agent = max(scores, key=scores.get)

    return {
        speaker: "representative" if speaker == agent else "customer"
        for speaker in speakers
    }


def format_segments_with_roles(segments, speaker_role_map):
    """Convert speaker-labeled segments to the model's customer/representative format."""
    lines = []
    for segment in segments or []:
        speaker = _speaker_label(segment)
        role = speaker_role_map.get(speaker, "customer")
        text = (segment.get("text") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


# Temsilcinin bekletme/araştırma anonsu ("sizi bekletiyorum, kontrol ediyorum"...).
HOLD_ANNOUNCE = (
    "bir saniye", "bir dakika", "kisa bir sure", "kontrol ediyorum", "kontrol edeyim",
    "kontrol sagliyorum", "hatta kalin", "hatta bekleyin", "hatta kalir misiniz",
    "sizi bekletiyorum", "bekletmem gerek", "bekletecegim", "arastiriyorum",
    "bakiyorum", "kontrol etmem gerek", "lutfen bekleyin", "sistemden bakiyorum",
)


def analyze_silence(segments, role_map=None, threshold=10.0):
    """WhisperX zaman damgalarından sessizlik/ölü hava istatistiği çıkarır.

    Ardışık konuşma parçaları arasındaki boşluk = o anki sessizlik. 'threshold'
    saniyeden uzun boşluklar 'ölü hava' sayılır. Her ölü hava olayı için,
    boşluktan HEMEN ÖNCEKİ replikte (temsilciye aitse) bekletme anonsu var mı
    diye bakılır; varsa olay 'announced' (uyarılı), yoksa 'unannounced' olur.

    Döner: dict | None
        Zaman damgası yoksa (ör. metin analizi) None döner.
    """
    timed = []
    for seg in segments or []:
        start = seg.get("start")
        end = seg.get("end")
        if start is None or end is None:
            continue
        timed.append({
            "start": float(start),
            "end": float(end),
            "speaker": _speaker_label(seg),
            "text": (seg.get("text") or ""),
        })

    if len(timed) < 2:
        return None

    timed.sort(key=lambda s: s["start"])

    total_silence = 0.0
    longest_gap = 0.0
    announced = 0
    unannounced = 0
    events = []

    for prev, nxt in zip(timed, timed[1:]):
        gap = nxt["start"] - prev["end"]
        if gap <= 0:
            continue
        total_silence += gap
        longest_gap = max(longest_gap, gap)
        if gap >= threshold:
            before_text = _fold(prev["text"])
            before_is_agent = (
                role_map is None
                or role_map.get(prev["speaker"]) == "representative"
            )
            is_announced = before_is_agent and any(h in before_text for h in HOLD_ANNOUNCE)
            if is_announced:
                announced += 1
            else:
                unannounced += 1
            events.append({
                "gap": round(gap, 1),
                "after_speaker": prev["speaker"],
                "announced": is_announced,
            })

    return {
        "threshold": threshold,
        "total_silence": round(total_silence, 1),
        "longest_gap": round(longest_gap, 1),
        "dead_air_count": announced + unannounced,
        "announced_count": announced,
        "unannounced_count": unannounced,
        "events": events,
    }
