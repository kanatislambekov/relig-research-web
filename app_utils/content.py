"""Pre-written text snippets used across the Streamlit pages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Section:
    """Simple container to organise extended abstract sections."""

    title: str
    paragraphs: List[str]


EXTENDED_ABSTRACT: List[Section] = [
    Section(
        title="Research Focus",
        paragraphs=[
            "The project investigates how religious affiliation and value systems "
            "shape fertility patterns in Kazakhstan. Using the national sample, "
            "the analysis contrasts Muslim and non-Muslim populations while "
            "controlling for socio-demographic, regional, and parental background "
            "characteristics.",
        ],
    ),
    Section(
        title="Fertility Differences by Denomination",
        paragraphs=[
            "Poisson regression models show that Muslim respondents have a higher "
            "number of children than non-Muslim peers even after accounting for "
            "controls. Two definitions of the Muslim population are evaluated: a "
            "conservative group that self-identifies as Muslim and an expanded "
            "group that additionally includes respondents listed as Hindu who share "
            "similar socio-demographic profiles. The fertility advantage persists "
            "for both groups, although effect sizes shrink slightly when controls "
            "are introduced.",
        ],
    ),
    Section(
        title="Family and Gender Value Indexes",
        paragraphs=[
            "Value-based indexes are constructed to capture expectations around "
            "family support (parental and children obligations) and egalitarian "
            "views about women's roles in society. Stronger family-support values "
            "are associated with higher fertility, whereas egalitarian values are "
            "linked to lower fertility. When the indexes are recoded into "
            "dichotomous variables, the negative influence of egalitarian values "
            "remains statistically significant. Muslim respondents display higher "
            "levels on family-oriented indexes, while both Muslim and non-Muslim "
            "groups exhibit reduced fertility with egalitarian orientations.",
        ],
    ),
    Section(
        title="Role of Religiosity",
        paragraphs=[
            "Self-reported religiosity modestly increases the number of children. "
            "A categorical specification, however, reveals a non-linear pattern in "
            "which any level of religiosity above the lowest category corresponds "
            "to higher fertility than among non-religious individuals. Interaction "
            "effects show that religiosity matters most within the Muslim "
            "population, especially among the most devout respondents, whereas "
            "differences among non-Muslim groups are negligible.",
        ],
    ),
    Section(
        title="Birth-Order Timing",
        paragraphs=[
            "Cox proportional hazards models explore the timing of first through "
            "fourth births. Muslim respondents maintain a higher probability of "
            "progressing to each birth order compared with non-Muslims across "
            "levels of religiosity. Family-support values accelerate the arrival of "
            "first and second births even after controlling for denomination, while "
            "egalitarian attitudes reduce the likelihood of early births for both "
            "religious groups.",
        ],
    ),
]

KEY_FINDINGS: List[str] = [
    "Muslim households consistently report larger completed fertility than "
    "non-Muslim households after accounting for demographic, regional, and "
    "background controls.",
    "Family-support norms raise fertility for both denominations, whereas "
    "egalitarian gender values correspond with lower fertility.",
    "The fertility premium associated with religiosity is concentrated among "
    "Muslim respondents, particularly at higher levels of devotion.",
    "Timing models confirm that Muslim respondents are more likely to reach the "
    "third birth order, the largest observed gap across groups.",
]
