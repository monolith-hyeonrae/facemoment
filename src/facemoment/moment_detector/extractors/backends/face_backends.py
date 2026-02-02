"""Face detection and expression analysis backend implementations.

This module re-exports backends from individual files for backward compatibility.
New code should import directly from the specific modules:

    from facemoment.moment_detector.extractors.backends.insightface import InsightFaceSCRFD
    from facemoment.moment_detector.extractors.backends.hsemotion import HSEmotionBackend
    from facemoment.moment_detector.extractors.backends.pyfeat import PyFeatBackend
"""

# Re-export for backward compatibility
from facemoment.moment_detector.extractors.backends.insightface import InsightFaceSCRFD
from facemoment.moment_detector.extractors.backends.hsemotion import HSEmotionBackend
from facemoment.moment_detector.extractors.backends.pyfeat import PyFeatBackend

__all__ = [
    "InsightFaceSCRFD",
    "HSEmotionBackend",
    "PyFeatBackend",
]
