"""Safety layer. Emergencies are never left to retrieval or an LLM: a fixed,
clinically conservative instruction is always shown first."""
import re
from typing import Optional
from .config import settings

_EMERGENCY = re.compile(
    r"(can'?t breathe|cannot breathe|trouble breathing|difficulty breathing|hard to breathe|wheez|"
    r"throat (is )?(closing|tight|swelling|swollen)|tongue (is )?swell|lips? (are )?swell|swelling (of|in) (my )?(face|lips|tongue|throat)|"
    r"anaphyla\w+ (now|right now|happening)|having (an )?(allergic reaction|anaphylaxis)|"
    r"(just|accidentally|think i) (ate|eaten|had)\b.*\b(allerg|nut|peanut|prawn|shellfish|egg|milk|sesame)|"
    r"feel (faint|dizzy)|passing out|collaps|going into shock|my (face|lips|throat|tongue) (is|are) (swelling|itchy|tingling))", re.I)

_SELF_HARM = re.compile(r"(kill myself|want to die|end my life|self[- ]harm|suicid)", re.I)


def emergency_banner(text: str) -> Optional[str]:
    if _EMERGENCY.search(text):
        return (f"**If you are having a reaction right now: use your adrenaline pen immediately, "
                f"call {settings.emergency_number}, and lie down with your legs raised (sit up if breathing is hard). "
                f"Do not wait to see if it gets better.**")
    return None


def wellbeing_message(text: str) -> Optional[str]:
    if _SELF_HARM.search(text):
        return ("I'm really sorry you're feeling this way. You deserve support from a real person right now - "
                "please talk to a trusted adult or contact a local crisis line (in Ireland: Samaritans 116 123, free, 24/7). "
                "If you're in immediate danger, call 112/999.")
    return None


DISCLAIMER = "Aahar.AI shares general education, not personal medical advice. Follow your allergy action plan and doctor's guidance."
