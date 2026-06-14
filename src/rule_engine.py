"""
Kural tabanlı çağrı kalite değerlendirme motoru.
====================================================================
BERTurk model skoru ANA skordur; bu motor deterministik kontroller yapıp
CEZA ve TAVAN uygular (proje tanımındaki hibrit yaklaşımın kural ayağı).

Mantık:
    nihai = model_skoru - toplam_ceza
    kritik ihlal (küfür, kaba dil) varsa nihai skora bir TAVAN uygulanır
    sonuç 0-100 aralığına sıkıştırılır

Her kontrol şeffaftır: hangi kriterin geçtiği/kaldığı ve kaç puan etkilediği
arayüzde döküm olarak gösterilebilir.

Ağırlıkları (ceza puanları) bu dosyanın üstündeki sabitlerden ayarlayabilirsin.
"""

import re

# ---------------------------------------------------------------------------
# CEZA PUANLARI (negatif etki). Ayarlanabilir.
# ---------------------------------------------------------------------------
PENALTY = {
    # Açılış / Selamlama
    "greeting": 3,
    "intro": 2,
    "offer_help": 3,
    # Kimlik Doğrulama
    "identity": 5,
    # Empati & Nezaket
    "empathy": 4,
    "empathy_when_negative": 7,  # müşteri olumsuzken empati yoksa daha ağır
    "courtesy": 2,
    # Çözüm & Kapanış
    "solution": 5,
    "closing_question": 2,
    "thanks_farewell": 2,
    # Anomali (ceza)
    "profanity": 40,
    "rude": 15,
    "shouting": 8,
    # Sessizlik / ölü hava (yalnızca ses analizinde; zaman damgası gerekir)
    "dead_air_unannounced": 10,  # habersiz uzun sessizlik (olay başına)
    "dead_air_announced": 3,     # anonslu uzun sessizlik (olay başına)
}

# Sessizlik cezasının üst sınırı (çok sayıda olayda skoru tamamen çökertmesin).
DEAD_AIR_PENALTY_CAP = 25

# ÖDÜL PUANLARI (pozitif etki). Geçen her pozitif kriter küçük bonus kazandırır;
# böylece tüm kriterleri geçen + model skoru yüksek bir çağrı 100'e ulaşabilir.
# Anomali kontrolleri (küfür yok vb.) bonus KAZANDIRMAZ — sadece ceza önler.
# Toplam ~14 puan; yani model ~86+ ve kusursuz uyum -> 100.
BONUS = {
    "greeting": 1.5,
    "intro": 1.0,
    "offer_help": 1.5,
    "identity": 2.0,
    "empathy": 2.0,
    "courtesy": 1.0,
    "solution": 2.5,
    "closing_question": 1.0,
    "thanks_farewell": 1.5,
}

# Anomali durumunda nihai skora uygulanan TAVAN (bunun üstüne çıkamaz).
CAP = {
    "profanity": 40,
    "rude": 60,
}

# ---------------------------------------------------------------------------
# ANAHTAR KELİME LİSTELERİ (accent-folded, küçük harf eşleşir)
# ---------------------------------------------------------------------------
GREETING = ["merhaba", "iyi gunler", "gunaydin", "iyi aksamlar", "hos geldiniz", "hosgeldiniz"]
INTRO = ["ben ", "adim ", "firma", "telekom", "musteri hizmetleri", "cagri merkezi"]
OFFER_HELP = ["nasil yardimci", "size nasil", "yardimci olabilirim", "buyurun"]
IDENTITY = ["musteri numara", "musteri no", "tc kimlik", "kimlik numara", "dogrulamak icin",
            "dogrulama", "hesabinizi kontrol", "numaranizi alabilir"]
EMPATHY = ["anliyorum", "haklisiniz", "uzgunum", "ozur dilerim", "cok iyi anliyorum",
           "sizi anliyorum", "gecmis olsun", "maalesef"]
COURTESY = ["rica ederim", "tabii", "memnuniyetle", "lutfen", "tesekkur ederim", "elbette"]
SOLUTION = ["talep olustur", "olusturuyorum", "duzelt", "duzeltme", "iletecegim", "ilettim",
            "kontrol ediyorum", "kontrol saglyorum", "iade", "cozum", "aktif ettim",
            "tanimladim", "guncelliyorum", "yonlendiriyorum", "kaydinizi aldim"]
CLOSING_QUESTION = ["baska bir konu", "baska bir sey", "yardimci olabilecegim baska",
                    "baska bir talebiniz"]
THANKS_FAREWELL = ["tesekkur ederim", "iyi gunler dileriz", "iyi gunler dilerim",
                   "iyi aksamlar dilerim", "bizi tercih", "saglikli gunler", "hosca kalin"]

# Müşterinin olumsuz duygu gösterdiği işaretler (empati bağlamı için)
NEGATIVE_CUSTOMER = ["sinirlendim", "sinir", "kizgin", "kizdim", "sikayet", "rezalet",
                     "berbat", "cok kotu", "biktim", "yeter artik", "memnun degil",
                     "canim sikildi", "ofkeli", "kabul edilemez"]

# Temsilcinin kaba/ilgisiz ifadeleri (ceza)
RUDE = ["ne yapayim yani", "elimden bir sey gelmez", "bana ne", " benim sucum degil",
        "kapatiyorum", "ugrasamam", "sizin probleminiz", "napayim", "bilmiyorum iste",
        "baska kanaldan deneyin"]

# Küfür/hakaret (kısa örnek liste — kendi listeni genişletebilirsin)
# Küfür / hakaret (fold edilmiş — Türkçe karakter sadeleştirilmiş hâlleriyle).
# Kelime-sınırı (\b) ile eşleşir; "sıkıntı", "normal" gibi masum kelimelerde
# yanlış pozitif olmaz. Kendi senaryolarına göre genişletebilirsin.
PROFANITY = [
    # hakaret / aşağılama
    "aptal", "salak", "gerizekali", "geri zekali", "mal misin", "sacmalama",
    "defol", "kapa ceneni", "ahmak", "dangalak", "embesil", "serefsiz",
    "haysiyetsiz", "alcak", "gerizekalisin",
    # açık küfür
    "orospu", "orospu cocugu", "pic", "pic kurusu", "siktir", "siktir git",
    "sikeyim", "sikerim", "amina", "aminako", "amcik", "yarrak", "yarak",
    "gavat", "kahpe", "pezevenk", "ananiki", "anani", "amk", "aq", "mk",
]


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def _fold(text):
    """Türkçe karakterleri sadeleştirip küçük harfe çevirir (accent-insensitive)."""
    if not text:
        return ""
    text = text.replace("İ", "i").replace("I", "i").replace("ı", "i")
    text = text.lower()
    for a, b in (("ç", "c"), ("ş", "s"), ("ğ", "g"), ("ü", "u"), ("ö", "o"), ("â", "a")):
        text = text.replace(a, b)
    return text


def _contains_any(text, keywords):
    return any(kw in text for kw in keywords)


_PROFANITY_RE = re.compile(
    r"\b(" + "|".join(re.escape(_fold(w)) for w in PROFANITY) + r")\b"
)


def _contains_profanity(text):
    """Küfür/hakareti kelime sınırıyla arar (yanlış pozitifsiz)."""
    return bool(_PROFANITY_RE.search(text or ""))


def _parse_transcript(transcript):
    """'representative: ...' / 'customer: ...' satırlarını role göre ayırır."""
    rep_lines, cust_lines, raw_rep_lines = [], [], []
    for line in (transcript or "").splitlines():
        if ":" not in line:
            continue
        role, _, text = line.partition(":")
        role = role.strip().lower()
        text = text.strip()
        if role in ("representative", "temsilci", "agent"):
            rep_lines.append(text)
            raw_rep_lines.append(text)
        elif role in ("customer", "musteri", "müşteri"):
            cust_lines.append(text)
    return rep_lines, cust_lines, raw_rep_lines


def _detect_shouting(raw_rep_lines):
    """Metin tabanlı bağırma sezgisi: çok sayıda BÜYÜK HARF bloğu veya '!!!'.
    Not: Whisper transkriptlerinde bu sınırlıdır; gerçek bağırma ses
    enerjisinden daha güvenilir tespit edilir."""
    joined = " ".join(raw_rep_lines)
    if "!!!" in joined:
        return True
    caps_words = re.findall(r"\b[A-ZÇŞĞÜÖİ]{3,}\b", joined)
    return len(caps_words) >= 2


# ---------------------------------------------------------------------------
# Ana değerlendirme
# ---------------------------------------------------------------------------
def evaluate_rules(transcript, model_score, silence_stats=None):
    """Transkripti kurallara göre değerlendirir ve nihai skoru hesaplar.

    silence_stats: formatting.analyze_silence çıktısı (yalnızca ses analizinde).
        None ise sessizlik/ölü hava kategorisi eklenmez (metin analizi).

    Döner: dict
        checks: [{category, label, passed, penalty}]
        penalty_total, applied_cap, model_score, final_score,
        passed_count, failed_count, silence
    """
    rep_lines, cust_lines, raw_rep_lines = _parse_transcript(transcript)
    rep = _fold(" ".join(rep_lines))
    cust = _fold(" ".join(cust_lines))

    checks = []

    def add(category, label, passed, penalty, bonus=0):
        passed = bool(passed)
        checks.append({
            "category": category,
            "label": label,
            "passed": passed,
            "penalty": 0 if passed else penalty,
            "bonus": bonus if passed else 0,
        })

    # 1) Opening / Greeting
    add("Opening", "Greeting given", _contains_any(rep, GREETING), PENALTY["greeting"], BONUS["greeting"])
    add("Opening", "Introduced company/self", _contains_any(rep, INTRO), PENALTY["intro"], BONUS["intro"])
    add("Opening", "Offered help", _contains_any(rep, OFFER_HELP), PENALTY["offer_help"], BONUS["offer_help"])

    # 2) Identity Verification
    add("Identity", "Identity/customer verification", _contains_any(rep, IDENTITY), PENALTY["identity"], BONUS["identity"])

    # 3) Empathy & Courtesy
    customer_negative = _contains_any(cust, NEGATIVE_CUSTOMER)
    has_empathy = _contains_any(rep, EMPATHY)
    empathy_penalty = PENALTY["empathy_when_negative"] if customer_negative else PENALTY["empathy"]
    empathy_label = ("Empathy shown (customer was upset)" if customer_negative
                     else "Empathy expression used")
    add("Empathy", empathy_label, has_empathy, empathy_penalty, BONUS["empathy"])
    add("Empathy", "Courtesy expressions", _contains_any(rep, COURTESY), PENALTY["courtesy"], BONUS["courtesy"])

    # 4) Resolution & Closing
    add("Resolution", "Offered concrete solution/action", _contains_any(rep, SOLUTION), PENALTY["solution"], BONUS["solution"])
    add("Closing", "Asked about further needs", _contains_any(rep, CLOSING_QUESTION), PENALTY["closing_question"], BONUS["closing_question"])
    add("Closing", "Thanks/proper farewell", _contains_any(rep, THANKS_FAREWELL), PENALTY["thanks_farewell"], BONUS["thanks_farewell"])

    # 5) Anomaly (penalty + cap) — passing gives no bonus, only avoids penalty.
    has_profanity = _contains_profanity(rep)
    has_rude = _contains_any(rep, RUDE)
    has_shouting = _detect_shouting(raw_rep_lines)
    add("Anomaly", "No profanity/insults", not has_profanity, PENALTY["profanity"])
    add("Anomaly", "No rude/dismissive language", not has_rude, PENALTY["rude"])
    add("Anomaly", "No shouting markers", not has_shouting, PENALTY["shouting"])

    # 6) Sessizlik / Ölü hava (yalnızca ses analizinde; zaman damgası gerekir)
    if silence_stats:
        unannounced = silence_stats.get("unannounced_count", 0)
        announced = silence_stats.get("announced_count", 0)
        dead_air = silence_stats.get("dead_air_count", 0)
        threshold = silence_stats.get("threshold", 10)
        silence_penalty = min(
            unannounced * PENALTY["dead_air_unannounced"]
            + announced * PENALTY["dead_air_announced"],
            DEAD_AIR_PENALTY_CAP,
        )
        if dead_air == 0:
            label = f"No long silence / dead air (>{threshold:g}s)"
        elif unannounced:
            label = (f"Unannounced dead air: {unannounced} event(s)"
                     + (f" (+{announced} announced)" if announced else ""))
        else:
            label = f"Announced hold: {announced} event(s) (minor)"
        add("Silence", label, dead_air == 0, silence_penalty)

    penalty_total = sum(c["penalty"] for c in checks)
    bonus_total = sum(c["bonus"] for c in checks)

    # Tavan: kritik ihlal varsa skora üst sınır
    caps = []
    if has_profanity:
        caps.append(CAP["profanity"])
    if has_rude:
        caps.append(CAP["rude"])
    applied_cap = min(caps) if caps else None

    # Nihai: model + ödül − ceza, sonra (varsa) tavan, sonra 0-100 sıkıştırma.
    final = model_score + bonus_total - penalty_total
    if applied_cap is not None:
        final = min(final, applied_cap)
    final = float(max(0.0, min(100.0, final)))

    passed_count = sum(1 for c in checks if c["passed"])
    failed_count = len(checks) - passed_count

    return {
        "checks": checks,
        "penalty_total": penalty_total,
        "bonus_total": round(bonus_total, 2),
        "applied_cap": applied_cap,
        "model_score": round(float(model_score), 2),
        "final_score": round(final, 2),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "silence": silence_stats,
    }
