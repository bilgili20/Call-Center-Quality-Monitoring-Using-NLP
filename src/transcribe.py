from shutil import which

import torch


_TORCH_LOAD_PATCHED = False


def _patch_torch_load_for_pyannote():
    """PyTorch 2.6+ uyumluluk yaması.

    PyTorch 2.6'da torch.load varsayılanı weights_only=True oldu. pyannote model
    checkpoint'leri omegaconf nesneleri içerdiğinden bu varsayılanla yükleme
    başarısız olur ("Unsupported global: omegaconf.listconfig.ListConfig").

    pyannote resmi/güvenilir bir kaynak olduğundan, çağrı sırasında weights_only
    açıkça verilmemişse False'a çekiyoruz. Yama yalnızca bir kez uygulanır.
    """
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return

    # Mümkünse güvenli yol: bilinen global'leri allowlist'e ekle.
    try:
        from omegaconf.listconfig import ListConfig
        from omegaconf.dictconfig import DictConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode

        torch.serialization.add_safe_globals(
            [ListConfig, DictConfig, ContainerMetadata, Metadata, AnyNode]
        )
    except Exception:
        pass

    # Geri dönüş: weights_only'i HER ZAMAN False'a zorla. Bazı kütüphaneler
    # (lightning / pyannote) torch.load'u açıkça weights_only=True ile çağırır;
    # setdefault bunu ezemediği için değeri doğrudan override ediyoruz.
    # (pyannote güvenilir kaynak olduğundan bu güvenli.)
    _original_load = getattr(torch, "_original_load_unpatched", torch.load)
    torch._original_load_unpatched = _original_load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load
    _TORCH_LOAD_PATCHED = True


def transcribe_with_whisperx(audio_path, hf_token=None, language="tr"):
    """
    Transcribe Turkish audio with WhisperX and optionally assign speaker labels.

    Diarization requires a Hugging Face token with access to the required
    pyannote models. When no token is provided, transcription still runs and
    segments may not include speaker labels.
    """
    if which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg bulunamadı. Ses dosyalarını işlemek için ffmpeg kurulmalıdır."
        )

    try:
        import whisperx
    except ImportError as exc:
        raise ImportError(
            "WhisperX içe aktarılamadı. Gerçek sebep: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    _patch_torch_load_for_pyannote()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # CPU'da hız için "medium" kullanılıyor (large-v3'ten ~3-4x daha hızlı,
    # Türkçede doğruluk farkı küçük). GPU'n varsa "large-v3"e çekebilirsin.
    model = whisperx.load_model(
        "medium",
        device=device,
        compute_type=compute_type,
        language=language,
    )
    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(audio, batch_size=8, language=language)

    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if hf_token:
        assign_word_speakers = _resolve_assign_word_speakers(whisperx)
        # m4a/mp3 gibi formatlarda pyannote dosyayı kendi okuyamaz; bu yüzden
        # whisperx'in zaten 16kHz mono olarak yüklediği 'audio' dizisini veriyoruz.
        diarize_segments = _run_pyannote_diarization(audio, hf_token, device)
        result = assign_word_speakers(diarize_segments, result)

    return result["segments"]


_HF_HUB_PATCHED = False


def _patch_hf_hub_use_auth_token():
    """Yeni huggingface_hub sürümlerinde kaldırılan 'use_auth_token' parametresini
    geri uyumlu hale getirir.

    Sorun: hf_hub_download / snapshot_download artık 'use_auth_token' kabul etmiyor
    (yeni adı 'token'). Kurulu pyannote ise indirme çağrılarında hâlâ
    'use_auth_token' kullanıyor → "got an unexpected keyword argument 'use_auth_token'".

    Çözüm: Bu fonksiyonları, 'use_auth_token' geldiğinde değeri 'token'a çeviren
    bir sarmalayıcıyla değiştiriyoruz. Hem huggingface_hub modülünde hem de bu
    fonksiyonları doğrudan import etmiş tüm modüllerde referansı güncelliyoruz.
    """
    global _HF_HUB_PATCHED
    if _HF_HUB_PATCHED:
        return

    import sys
    import functools
    import huggingface_hub

    def _wrap(original):
        if getattr(original, "_uat_wrapped", False):
            return original

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            if "use_auth_token" in kwargs:
                token = kwargs.pop("use_auth_token")
                kwargs.setdefault("token", token)
            return original(*args, **kwargs)

        wrapper._uat_wrapped = True
        return wrapper

    target_names = ("hf_hub_download", "snapshot_download")
    originals = {}
    for name in target_names:
        orig = getattr(huggingface_hub, name, None)
        if orig is None:
            continue
        wrapped = _wrap(orig)
        originals[orig] = wrapped
        setattr(huggingface_hub, name, wrapped)

    # `from huggingface_hub import hf_hub_download` ile doğrudan import etmiş
    # modüllerdeki referansları da güncelle.
    for module in list(sys.modules.values()):
        if module is None:
            continue
        for name in target_names:
            ref = getattr(module, name, None)
            if ref in originals:
                try:
                    setattr(module, name, originals[ref])
                except Exception:
                    pass

    _HF_HUB_PATCHED = True


def _run_pyannote_diarization(audio, hf_token, device):
    """pyannote.audio ile doğrudan diarization çalıştırır ve WhisperX'in
    assign_word_speakers fonksiyonunun beklediği DataFrame'i döndürür.

    'audio': whisperx.load_audio'nun döndürdüğü 16kHz mono float32 numpy dizisi.
    Dosya yolu yerine bu diziyi kullanıyoruz; böylece pyannote'un m4a/mp3 gibi
    formatları kendi okuyamaması sorununu (soundfile m4a desteklemez) tamamen
    aşıyoruz — ffmpeg ile çözme işini zaten whisperx yapmış oluyor.

    WhisperX'in kendi DiarizationPipeline sarmalayıcısı eski olduğundan token'ı
    yeni huggingface_hub'ın kabul etmediği 'use_auth_token' adıyla iletiyor.
    Bunu atlamak için önce HF'ye login olup, modeli token parametresi GEÇMEDEN
    yüklüyoruz; böylece parametre-adı uyumsuzluğu tamamen ortadan kalkıyor.
    """
    try:
        import pandas as pd
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ImportError(
            "pyannote.audio yüklü değil. Kurulum: pip install pyannote.audio"
        ) from exc

    # 0) huggingface_hub uyumluluk yaması: yeni hub sürümlerinde hf_hub_download /
    #    snapshot_download artık 'use_auth_token' kabul etmiyor; pyannote ise hâlâ
    #    bu eski adla çağırıyor. Çağrıyı yakalayıp 'token'a çeviriyoruz.
    _patch_hf_hub_use_auth_token()

    # 1) Token'ı önbelleğe al; from_pretrained'e kwarg geçmeye gerek kalmasın.
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
    except Exception:
        pass

    # 2) Pipeline'ı yükle. pyannote'un from_pretrained'i sadece 'use_auth_token'
    #    parametresini tanır ('token' değil). Auth/lisans başarısızsa hata
    #    fırlatmaz, sessizce None döner — bunu yakalayıp net mesaj veriyoruz.
    model_name = "pyannote/speaker-diarization-3.1"
    pipeline = None
    errors = []
    for kwargs in ({"use_auth_token": hf_token}, {}):
        try:
            pipeline = Pipeline.from_pretrained(model_name, **kwargs)
            if pipeline is not None:
                break
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue

    if pipeline is None:
        detail = (" Teknik detay: " + " | ".join(errors)) if errors else ""
        raise RuntimeError(
            "pyannote diarization modeli yüklenemedi. Bu neredeyse her zaman "
            "TOKEN veya LİSANS sorunudur (kod sorunu değil):\n"
            "  1) https://hf.co/pyannote/speaker-diarization-3.1 sayfasına gir ve "
            "koşulları kabul et (Agree).\n"
            "  2) https://hf.co/pyannote/segmentation-3.0 sayfasında da kabul et.\n"
            "  3) Token'ın 'read' yetkili ve doğru hesaba ait olduğundan emin ol.\n"
            "İkisini de kabul ettikten sonra uygulamayı yeniden başlat." + detail
        )

    # 3) Mümkünse GPU'ya taşı.
    try:
        pipeline.to(torch.device(device))
    except Exception:
        pass

    # 4) Sesi pyannote'un beklediği {waveform, sample_rate} formatına çevir.
    #    16kHz mono numpy -> (1, zaman) şeklinde torch tensor.
    import numpy as np

    waveform = torch.as_tensor(np.asarray(audio, dtype="float32"))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, zaman) = tek kanal

    diarization = pipeline({"waveform": waveform, "sample_rate": 16000})
    rows = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    return pd.DataFrame(rows, columns=["start", "end", "speaker"])


def _resolve_assign_word_speakers(whisperx):
    """assign_word_speakers fonksiyonunu sürümden bağımsız bulur."""
    if hasattr(whisperx, "assign_word_speakers"):
        return whisperx.assign_word_speakers
    try:
        from whisperx.diarize import assign_word_speakers
        return assign_word_speakers
    except ImportError as exc:
        raise AttributeError(
            "assign_word_speakers fonksiyonu bu WhisperX sürümünde bulunamadı."
        ) from exc
