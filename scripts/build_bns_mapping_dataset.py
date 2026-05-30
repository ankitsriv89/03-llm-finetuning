"""
build_bns_mapping_dataset.py — Generate IPC/CrPC/IEA → BNS/BNSS/BSA training data
==================================================================================
Generates a synthetic-but-authoritative dataset teaching the model the mapping
between India's repealed colonial-era codes and the new post-July-2024 codes.

Output: data/bns_bnss_bsa_mapping.jsonl  (instruction-tuning format)

Why this dataset is needed:
    The viber1/indian-law-dataset and Exploration-Lab/InLegalNLI datasets used in
    Phase 3b were both built before July 2024. They teach the model IPC §302 as
    current law — but as of 2026, IPC has been REPEALED. The current statute for
    murder is BNS §103. Without this dataset, the fine-tuned model will confidently
    cite repealed sections in production — a critical failure for JusticeAI.

Data sources (all authoritative, publicly available):
    - Ministry of Law BNS/BNSS/BSA gazette (2023)
    - LiveLaw new criminal laws comparison
    - Bar Council of India transition guides
    - SCC Online "New Criminal Laws" microsite

Mapping coverage (seed mappings, hand-curated for accuracy):
    - BNS:  ~60 most-litigated IPC → BNS mappings (murder, theft, rape, cheating, etc.)
    - BNSS: ~30 CrPC → BNSS procedural mappings (bail, FIR, arrest, charge framing)
    - BSA:  ~20 IEA → BSA evidentiary mappings (electronic records, hearsay, expert evidence)

Generation strategy:
    For each seed mapping, we generate 5–8 question variants (forward lookup, reverse
    lookup, punishment, procedure, contextual application) → ~500–800 total samples.
    This gives the model enough exposure to learn the mapping in multiple framings.

Usage:
    python scripts/build_bns_mapping_dataset.py
    # outputs: data/bns_bnss_bsa_mapping.jsonl
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


# ─────────────────────────────────────────────
# Seed mapping data
# ─────────────────────────────────────────────
# Each tuple: (old_code, old_section, new_code, new_section, offense/topic,
#              punishment_or_procedure, key_change_note)

@dataclass(frozen=True)
class Mapping:
    old_code: str           # "IPC" / "CrPC" / "IEA"
    old_section: str        # "302" / "437" / "65B"
    new_code: str           # "BNS" / "BNSS" / "BSA"
    new_section: str        # "103" / "480" / "63"
    topic: str              # short label, e.g. "Murder"
    description: str        # one-sentence description of the offense/procedure
    punishment: str         # punishment text (for criminal offenses; "—" for procedural)
    change_note: str        # what's notably new vs old


# IPC → BNS (criminal offenses)
# Source: Ministry of Law gazette + LiveLaw comparison tables.
# Coverage focuses on most-litigated provisions — sufficient for bail/criminal QA.
BNS_MAPPINGS: list[Mapping] = [
    Mapping("IPC", "302", "BNS", "103", "Murder",
            "Unlawful killing with intention or knowledge.",
            "Death or imprisonment for life, and fine.",
            "Section number changed; punishment unchanged. BNS §103(2) adds mob lynching as a specific offense."),
    Mapping("IPC", "304", "BNS", "105", "Culpable homicide not amounting to murder",
            "Causing death without the intention/knowledge required for murder.",
            "Imprisonment for life, or up to 10 years (Part I); up to 10 years (Part II).",
            "Section renumbered. Definition substantively unchanged."),
    Mapping("IPC", "304A", "BNS", "106", "Causing death by negligence",
            "Causing death by rash or negligent act not amounting to culpable homicide.",
            "Up to 5 years (general); BNS §106(2) introduces up to 10 years for hit-and-run where driver flees without reporting.",
            "BNS §106(2) is NEW — hit-and-run with flight has been carved out as an aggravated offense."),
    Mapping("IPC", "304B", "BNS", "80", "Dowry death",
            "Death of woman within 7 years of marriage in connection with dowry demand.",
            "Imprisonment for not less than 7 years, extendable to life.",
            "Section renumbered; provisions retained."),
    Mapping("IPC", "306", "BNS", "108", "Abetment of suicide",
            "Abetting the commission of suicide.",
            "Up to 10 years and fine.",
            "Renumbered. BNS retains the same constitutional concerns flagged by SC in Gurcharan Singh."),
    Mapping("IPC", "307", "BNS", "109", "Attempt to murder",
            "Doing an act with intention or knowledge to commit murder.",
            "Up to 10 years; if hurt caused, up to life; if accused is life-convict, may extend to death.",
            "Renumbered. Punishment structure retained."),
    Mapping("IPC", "323", "BNS", "115(2)", "Voluntarily causing hurt",
            "Causing bodily pain, disease, or infirmity voluntarily.",
            "Up to 1 year or fine up to ₹10,000 or both.",
            "Renumbered as sub-section. Fine amount updated from ₹1,000 to ₹10,000."),
    Mapping("IPC", "324", "BNS", "118(1)", "Voluntarily causing hurt by dangerous weapons",
            "Hurt caused using weapons or means likely to cause death.",
            "Up to 3 years or fine up to ₹20,000 or both.",
            "Renumbered as sub-section. Fine updated."),
    Mapping("IPC", "325", "BNS", "117(2)", "Voluntarily causing grievous hurt",
            "Causing grievous hurt voluntarily.",
            "Up to 7 years and fine.",
            "Renumbered as sub-section."),
    Mapping("IPC", "326", "BNS", "118(2)", "Voluntarily causing grievous hurt by dangerous weapons",
            "Grievous hurt caused using weapons or means likely to cause death.",
            "Up to life or up to 10 years and fine.",
            "Renumbered as sub-section."),
    Mapping("IPC", "354", "BNS", "74", "Assault to outrage modesty of woman",
            "Assault or criminal force on a woman with intent to outrage her modesty.",
            "1 to 5 years and fine.",
            "Renumbered. Provisions retained."),
    Mapping("IPC", "354A", "BNS", "75", "Sexual harassment",
            "Physical contact, unwelcome advances, demand for sexual favours.",
            "Up to 3 years or fine or both (depending on sub-clause).",
            "Renumbered. Provisions retained."),
    Mapping("IPC", "354B", "BNS", "76", "Assault to disrobe woman",
            "Assault or use of criminal force to disrobe a woman.",
            "3 to 7 years and fine.",
            "Renumbered. Provisions retained."),
    Mapping("IPC", "354C", "BNS", "77", "Voyeurism",
            "Watching or capturing image of woman in private act.",
            "1 to 3 years (first conviction); 3 to 7 years (subsequent).",
            "Renumbered. Provisions retained."),
    Mapping("IPC", "354D", "BNS", "78", "Stalking",
            "Following or contacting a woman despite disinterest, or monitoring her electronic communications.",
            "Up to 3 years (first); up to 5 years (subsequent), and fine.",
            "Renumbered. Provisions retained."),
    Mapping("IPC", "375", "BNS", "63", "Rape — definition",
            "Definition of rape (acts constituting rape).",
            "— (definitional section)",
            "Renumbered. Definition substantively retained, with consent provisions clarified."),
    Mapping("IPC", "376", "BNS", "64", "Rape — punishment",
            "Punishment for rape.",
            "Not less than 10 years, extendable to life, and fine.",
            "Renumbered. Minimum punishment retained."),
    Mapping("IPC", "376A", "BNS", "66", "Rape causing death or persistent vegetative state",
            "Rape that causes death or leaves victim in PVS.",
            "Not less than 20 years; may extend to life or death.",
            "Renumbered."),
    Mapping("IPC", "376AB", "BNS", "65(2)", "Rape of woman under 12",
            "Rape of a woman under 12 years.",
            "Not less than 20 years, may extend to life (remainder of natural life), or death.",
            "Renumbered."),
    Mapping("IPC", "376D", "BNS", "70", "Gang rape",
            "Rape committed by one or more persons constituting a group.",
            "Not less than 20 years, may extend to life (remainder of natural life), and fine.",
            "Renumbered. Provisions retained."),
    Mapping("IPC", "377", "BNS", "—", "Unnatural offences",
            "Carnal intercourse against the order of nature.",
            "Decriminalised for consenting adults (Navtej Singh Johar, 2018).",
            "BNS has NO direct equivalent to IPC §377 for consensual adult acts. Non-consensual acts addressed via rape/sexual assault provisions."),
    Mapping("IPC", "378", "BNS", "303", "Theft — definition",
            "Definition of theft.",
            "— (definitional)",
            "Renumbered."),
    Mapping("IPC", "379", "BNS", "303(2)", "Theft — punishment",
            "Punishment for theft.",
            "Up to 3 years or fine or both.",
            "Renumbered as sub-section. BNS introduces community service for petty theft (value ≤ ₹5,000, first offence)."),
    Mapping("IPC", "380", "BNS", "305", "Theft in dwelling house",
            "Theft in a building used as a dwelling.",
            "Up to 7 years and fine.",
            "Renumbered."),
    Mapping("IPC", "392", "BNS", "309", "Robbery",
            "Theft or extortion with violence or fear of violence.",
            "Up to 10 years and fine; up to 14 years if on highway after sunset/before sunrise.",
            "Renumbered."),
    Mapping("IPC", "395", "BNS", "310", "Dacoity",
            "Robbery committed by five or more persons.",
            "Up to life or up to 10 years and fine.",
            "Renumbered."),
    Mapping("IPC", "405", "BNS", "316", "Criminal breach of trust — definition",
            "Dishonest misappropriation of property entrusted.",
            "— (definitional)",
            "Renumbered."),
    Mapping("IPC", "406", "BNS", "316(2)", "Criminal breach of trust — punishment",
            "Punishment for criminal breach of trust.",
            "Up to 5 years or fine or both.",
            "Renumbered as sub-section."),
    Mapping("IPC", "415", "BNS", "318", "Cheating — definition",
            "Definition of cheating.",
            "— (definitional)",
            "Renumbered."),
    Mapping("IPC", "420", "BNS", "318(4)", "Cheating and dishonestly inducing delivery of property",
            "Cheating with dishonest inducement.",
            "Up to 7 years and fine.",
            "Renumbered as sub-section under §318."),
    Mapping("IPC", "498A", "BNS", "85", "Cruelty by husband or relatives",
            "Cruelty by husband or his relatives towards a married woman.",
            "Up to 3 years and fine.",
            "Renumbered. Provisions retained."),
    Mapping("IPC", "499", "BNS", "356(1)", "Defamation — definition",
            "Definition of defamation.",
            "— (definitional)",
            "Renumbered as sub-section."),
    Mapping("IPC", "500", "BNS", "356(2)", "Defamation — punishment",
            "Punishment for defamation.",
            "Up to 2 years or fine or community service.",
            "Renumbered as sub-section. Community service is a NEW alternative under BNS — first time in Indian criminal law."),
    Mapping("IPC", "120A", "BNS", "61(1)", "Criminal conspiracy — definition",
            "Definition of criminal conspiracy.",
            "— (definitional)",
            "Renumbered."),
    Mapping("IPC", "120B", "BNS", "61(2)", "Criminal conspiracy — punishment",
            "Punishment for criminal conspiracy.",
            "Same as for the offence conspired to be committed.",
            "Renumbered. Provisions retained."),
    Mapping("IPC", "124A", "BNS", "152", "Sedition (now: 'Acts endangering sovereignty, unity and integrity of India')",
            "Acts threatening the sovereignty, unity, or integrity of India.",
            "Up to life or up to 7 years, and fine.",
            "MATERIAL CHANGE: 'Sedition' as a term is GONE. BNS §152 reframes it as acts endangering sovereignty. Subjective tests narrowed per SC's pending review (Kedar Nath continuing scrutiny)."),
    Mapping("IPC", "153A", "BNS", "196", "Promoting enmity between groups",
            "Promoting enmity on grounds of religion, race, language, etc.",
            "Up to 3 years or fine or both; if in place of worship, up to 5 years.",
            "Renumbered."),
    Mapping("IPC", "295A", "BNS", "299", "Deliberate acts to outrage religious feelings",
            "Insulting religion or religious beliefs of any class.",
            "Up to 3 years or fine or both.",
            "Renumbered."),
    Mapping("IPC", "—", "BNS", "111", "Organised crime — NEW",
            "Continuing unlawful activity by organized crime syndicate (extortion, kidnapping, contract killing, cyber-crime, trafficking).",
            "Up to life or death (if results in death); minimum 5 years otherwise.",
            "ENTIRELY NEW offense — no IPC equivalent. Replaces MCOCA-type provisions at central level."),
    Mapping("IPC", "—", "BNS", "112", "Petty organised crime — NEW",
            "Vehicle theft, snatching, shoplifting, ticket scalping by organised gangs.",
            "Imprisonment 1 to 7 years and fine.",
            "ENTIRELY NEW — created to address bike-snatching, ATM card theft gangs, etc."),
    Mapping("IPC", "—", "BNS", "113", "Terrorist act — NEW",
            "Acts with intent to threaten unity, integrity, sovereignty, security, or strike terror.",
            "Death or life imprisonment (if results in death); 5 years to life otherwise.",
            "Defined IN BNS itself (previously only in UAPA). Concurrent jurisdiction with UAPA — investigating officer decides which to apply."),
    Mapping("IPC", "—", "BNS", "69", "Sexual intercourse by deceitful means — NEW",
            "Sexual intercourse by deceit (promise to marry without intention, false identity, false promise of employment/promotion).",
            "Up to 10 years and fine.",
            "ENTIRELY NEW. Addresses 'false promise of marriage' cases that previously had no clean statutory home."),
    Mapping("IPC", "—", "BNS", "117(4)", "Grievous hurt causing PVS — NEW",
            "Grievous hurt that puts victim in permanent vegetative state or permanent disability.",
            "Not less than 10 years, may extend to life.",
            "ENTIRELY NEW sub-section."),
    Mapping("IPC", "—", "BNS", "150", "Acts endangering sovereignty — NEW framing",
            "See BNS §152 (sedition rework).",
            "See §152.",
            "Note: BNS §150 was reserved/renumbered during drafting; final sedition replacement landed at §152."),
    Mapping("IPC", "153B", "BNS", "197", "Imputations prejudicial to national integration",
            "Statements prejudicial to national integration.",
            "Up to 3 years or fine or both.",
            "Renumbered."),
    Mapping("IPC", "201", "BNS", "238", "Causing disappearance of evidence",
            "Causing disappearance of evidence of offence, or giving false information.",
            "Punishment depends on the underlying offense (up to 7 years if underlying is capital).",
            "Renumbered."),
    Mapping("IPC", "212", "BNS", "249", "Harbouring offender",
            "Harbouring a person known to have committed an offence.",
            "Up to 5 years (if underlying capital); up to 3 years (others).",
            "Renumbered."),
    Mapping("IPC", "279", "BNS", "281", "Rash driving",
            "Rash or negligent driving on a public way.",
            "Up to 6 months or fine up to ₹1,000 or both.",
            "Renumbered."),
    Mapping("IPC", "294", "BNS", "296", "Obscene acts in public",
            "Obscene acts or words in any public place.",
            "Up to 3 months or fine up to ₹1,000 or both.",
            "Renumbered."),
    Mapping("IPC", "337", "BNS", "125(a)", "Causing hurt by act endangering life",
            "Hurt caused by act endangering life or personal safety.",
            "Up to 6 months or fine up to ₹2,500 or both.",
            "Renumbered as sub-section."),
    Mapping("IPC", "338", "BNS", "125(b)", "Causing grievous hurt by act endangering life",
            "Grievous hurt caused by act endangering life.",
            "Up to 2 years or fine up to ₹5,000 or both.",
            "Renumbered as sub-section."),
    Mapping("IPC", "339", "BNS", "126", "Wrongful restraint",
            "Voluntarily obstructing a person.",
            "Up to 1 month or fine up to ₹5,000 or both.",
            "Renumbered."),
    Mapping("IPC", "340", "BNS", "127", "Wrongful confinement",
            "Wrongful restraint within certain limits.",
            "Up to 1 year or fine up to ₹5,000 or both.",
            "Renumbered."),
    Mapping("IPC", "359", "BNS", "135", "Kidnapping — definition",
            "Two kinds: kidnapping from India and kidnapping from lawful guardianship.",
            "— (definitional)",
            "Renumbered."),
    Mapping("IPC", "363", "BNS", "139", "Kidnapping — punishment",
            "Punishment for kidnapping.",
            "Up to 7 years and fine.",
            "Renumbered."),
    Mapping("IPC", "364A", "BNS", "140(2)", "Kidnapping for ransom",
            "Kidnapping for ransom.",
            "Death or imprisonment for life and fine.",
            "Renumbered as sub-section."),
    Mapping("IPC", "370", "BNS", "143", "Trafficking of persons",
            "Trafficking of persons.",
            "7 to 10 years and fine (basic); 10 years to life (aggravated).",
            "Renumbered."),
    Mapping("IPC", "415", "BNS", "318(1)", "Cheating",
            "Cheating defined.",
            "— (definitional in §318(1))",
            "See entry for IPC §420 → BNS §318(4) for punishment."),
    Mapping("IPC", "463", "BNS", "335", "Forgery — definition",
            "Making false document with intent to defraud or harm.",
            "— (definitional)",
            "Renumbered."),
    Mapping("IPC", "465", "BNS", "336(2)", "Forgery — punishment",
            "Punishment for forgery.",
            "Up to 2 years or fine or both.",
            "Renumbered as sub-section."),
    Mapping("IPC", "468", "BNS", "336(3)", "Forgery for cheating",
            "Forgery for purpose of cheating.",
            "Up to 7 years and fine.",
            "Renumbered as sub-section."),
]


# CrPC → BNSS (procedural)
BNSS_MAPPINGS: list[Mapping] = [
    Mapping("CrPC", "41", "BNSS", "35", "Power of police to arrest without warrant",
            "When police may arrest without warrant.",
            "— (procedural)",
            "Renumbered. Arnesh Kumar safeguards (notice under §41A CrPC → §35(7) BNSS, mandatory for offenses ≤ 7 years)."),
    Mapping("CrPC", "41A", "BNSS", "35(7)", "Notice of appearance before police",
            "Notice of appearance instead of arrest for offenses ≤ 7 years.",
            "— (procedural)",
            "Renumbered. Procedure retained per Arnesh Kumar guidelines."),
    Mapping("CrPC", "50", "BNSS", "47", "Person arrested to be informed of grounds and right to bail",
            "Arrested person must be informed of grounds and bail right.",
            "— (procedural)",
            "Renumbered. Article 22 constitutional requirement retained."),
    Mapping("CrPC", "57", "BNSS", "58", "24-hour rule",
            "No arrested person to be detained beyond 24 hours without Magistrate.",
            "— (procedural)",
            "Renumbered. Article 22(2) constitutional requirement retained."),
    Mapping("CrPC", "154", "BNSS", "173", "Information in cognizable cases (FIR)",
            "Recording of FIR.",
            "— (procedural)",
            "Renumbered. BNSS §173(1) NEW: FIR can be filed electronically (zero-FIR formalised); §173(3) allows preliminary enquiry for offenses 3-7 years before FIR."),
    Mapping("CrPC", "161", "BNSS", "180", "Examination of witnesses by police",
            "Police examination of witnesses during investigation.",
            "— (procedural)",
            "Renumbered. BNSS adds audio-video recording option for statements."),
    Mapping("CrPC", "164", "BNSS", "183", "Recording of confessions and statements",
            "Magistrate recording of confessions.",
            "— (procedural)",
            "Renumbered. BNSS mandates audio-video recording for statements of sexual offence victims."),
    Mapping("CrPC", "167", "BNSS", "187", "Procedure when investigation cannot be completed in 24 hours",
            "Judicial custody / police custody during investigation.",
            "— (procedural)",
            "Renumbered. MATERIAL CHANGE in BNSS §187(3): police custody can now be sought in parts up to 15 days within the first 40/60 days (not just first 15) — significantly expands custodial interrogation window. Subject to SC scrutiny."),
    Mapping("CrPC", "173", "BNSS", "193", "Police report on completion of investigation (chargesheet)",
            "Filing of chargesheet.",
            "— (procedural)",
            "Renumbered. BNSS §193(8): chargesheet within 90 days; extension only up to 90 more days."),
    Mapping("CrPC", "190", "BNSS", "210", "Cognizance of offenses by Magistrates",
            "Magistrate taking cognizance.",
            "— (procedural)",
            "Renumbered."),
    Mapping("CrPC", "204", "BNSS", "227", "Issue of process",
            "Magistrate issuing summons/warrant.",
            "— (procedural)",
            "Renumbered."),
    Mapping("CrPC", "227", "BNSS", "250", "Discharge",
            "Discharge of accused at framing of charge.",
            "— (procedural)",
            "Renumbered. BNSS §250(1) NEW: discharge application must be filed within 60 days of commitment."),
    Mapping("CrPC", "228", "BNSS", "251", "Framing of charge",
            "Framing of charge in Sessions trial.",
            "— (procedural)",
            "Renumbered."),
    Mapping("CrPC", "239", "BNSS", "262", "Discharge in warrant cases instituted on police report",
            "Discharge in warrant cases.",
            "— (procedural)",
            "Renumbered."),
    Mapping("CrPC", "313", "BNSS", "351", "Power to examine the accused",
            "Court examining the accused after prosecution evidence.",
            "— (procedural)",
            "Renumbered. BNSS §351(5) NEW: examination can be done by audio-video means."),
    Mapping("CrPC", "320", "BNSS", "359", "Compounding of offenses",
            "Compoundable offenses table.",
            "— (procedural)",
            "Renumbered. List of compoundable offenses updated to reflect BNS section numbers."),
    Mapping("CrPC", "374", "BNSS", "415", "Appeals from convictions",
            "Appeals from convictions by Sessions/HC.",
            "— (procedural)",
            "Renumbered."),
    Mapping("CrPC", "437", "BNSS", "480", "Bail in bailable / non-bailable offenses (Magistrate)",
            "Bail by Magistrate in non-bailable offenses.",
            "— (procedural)",
            "Renumbered. BNSS §480(6) NEW: first-time offender on undertrial detention exceeding 1/3rd of max punishment SHALL be released on bond (codifies §436A CrPC)."),
    Mapping("CrPC", "438", "BNSS", "482", "Anticipatory bail",
            "Pre-arrest anticipatory bail.",
            "— (procedural)",
            "Renumbered. Sushila Aggarwal protections retained."),
    Mapping("CrPC", "439", "BNSS", "483", "Special powers of HC/Sessions regarding bail",
            "Bail by HC or Sessions in non-bailable offenses.",
            "— (procedural)",
            "Renumbered. Provisions retained."),
    Mapping("CrPC", "436A", "BNSS", "479", "Maximum period of detention of undertrial",
            "Undertrial detention cap at half of max punishment (or 1/3rd for first-time offender — NEW).",
            "— (procedural)",
            "BNSS §479: first-time offender released after 1/3rd of max punishment (NEW relief). Excluded: offenses punishable by death/life."),
    Mapping("CrPC", "468", "BNSS", "514", "Bar to taking cognizance after lapse of period of limitation",
            "Limitation period for taking cognizance.",
            "— (procedural)",
            "Renumbered."),
    Mapping("CrPC", "482", "BNSS", "528", "Inherent powers of High Court",
            "HC inherent powers to prevent abuse of process.",
            "— (procedural)",
            "Renumbered. Provisions retained."),
    Mapping("CrPC", "164A", "BNSS", "184", "Medical examination of rape victim",
            "Mandatory medical examination of rape victim within 24 hours.",
            "— (procedural)",
            "Renumbered. Provisions retained."),
    Mapping("CrPC", "53A", "BNSS", "52", "Examination of person accused of rape by medical practitioner",
            "DNA / medical examination of rape accused.",
            "— (procedural)",
            "Renumbered."),
    Mapping("CrPC", "—", "BNSS", "356", "Trial in absentia — NEW",
            "Trial of proclaimed offender in absentia.",
            "— (procedural)",
            "ENTIRELY NEW. BNSS §356 allows trial in absentia for proclaimed offenders after 90-day notice + safeguards. Defense counsel appointed by State."),
    Mapping("CrPC", "—", "BNSS", "37", "Designated police officer — NEW",
            "Each district to designate police officer responsible for arrest information.",
            "— (procedural)",
            "ENTIRELY NEW. Designated officer must maintain digital register of arrests; visible to relatives."),
    Mapping("CrPC", "—", "BNSS", "172", "Day-to-day register of investigation diary — DIGITAL",
            "Case diary in electronic form.",
            "— (procedural)",
            "BNSS mandates digitisation of case diary (previously paper-only)."),
    Mapping("CrPC", "—", "BNSS", "176(3)", "Forensic team mandatory for offenses ≥ 7 years — NEW",
            "Forensic team visit to crime scene mandatory for offenses punishable ≥ 7 years.",
            "— (procedural)",
            "ENTIRELY NEW. Major procedural reform aimed at strengthening evidence quality."),
    Mapping("CrPC", "—", "BNSS", "530", "Electronic mode for proceedings — NEW",
            "All trials, inquiries, proceedings may be held in electronic mode.",
            "— (procedural)",
            "ENTIRELY NEW. Codifies COVID-era video conferencing; e-FIR, e-summons, e-trial enabled platform-wide."),
]


# IEA → BSA (evidence)
BSA_MAPPINGS: list[Mapping] = [
    Mapping("IEA", "3", "BSA", "2", "Interpretation clause — definitions",
            "Definitions including 'evidence', 'proved', 'document'.",
            "— (definitional)",
            "Renumbered. 'Document' now expressly includes electronic and digital records (§2(d))."),
    Mapping("IEA", "17", "BSA", "15", "Admission — definition",
            "Definition of admission.",
            "— (definitional)",
            "Renumbered."),
    Mapping("IEA", "24", "BSA", "22", "Confession caused by inducement, threat, promise",
            "Confessions tainted by inducement/threat are irrelevant.",
            "— (rule of evidence)",
            "Renumbered. Provisions retained."),
    Mapping("IEA", "25", "BSA", "23", "Confession to police officer",
            "Confession to police officer is inadmissible.",
            "— (rule of evidence)",
            "Renumbered. Provisions retained per Aghnoo Nagesia jurisprudence."),
    Mapping("IEA", "26", "BSA", "23(2)", "Confession in police custody",
            "Confession in custody, unless made in presence of Magistrate, is inadmissible.",
            "— (rule of evidence)",
            "Renumbered as sub-section."),
    Mapping("IEA", "27", "BSA", "23(2) proviso", "Statement leading to discovery",
            "Statement leading to discovery of fact is admissible to that extent.",
            "— (rule of evidence)",
            "Renumbered. Pulukuri Kotayya jurisprudence applies."),
    Mapping("IEA", "32", "BSA", "26", "Statements by persons who cannot be called as witnesses (dying declaration)",
            "Statements by dead/unavailable persons (dying declarations).",
            "— (rule of evidence)",
            "Renumbered. Dying declaration jurisprudence (Kushal Rao) retained."),
    Mapping("IEA", "45", "BSA", "39", "Opinion of experts",
            "Expert opinion on foreign law, science, art, identity of handwriting/fingerprints.",
            "— (rule of evidence)",
            "Renumbered."),
    Mapping("IEA", "59", "BSA", "53", "Proof of facts by oral evidence",
            "All facts (except contents of documents/electronic records) may be proved by oral evidence.",
            "— (rule of evidence)",
            "Renumbered. Electronic records now explicitly carved out."),
    Mapping("IEA", "61", "BSA", "56", "Proof of contents of documents",
            "Documents proved by primary or secondary evidence.",
            "— (rule of evidence)",
            "Renumbered."),
    Mapping("IEA", "65A", "BSA", "62", "Special provisions as to evidence relating to electronic record",
            "Electronic records evidence governed by §63 (was §65B IEA).",
            "— (rule of evidence)",
            "Renumbered."),
    Mapping("IEA", "65B", "BSA", "63", "Admissibility of electronic records",
            "Conditions for admitting electronic records — certificate requirement.",
            "— (rule of evidence)",
            "MATERIAL: renumbered, and BSA §63 EXPANDS scope to include semiconductor memory, cloud storage, communications devices. Certificate format updated."),
    Mapping("IEA", "85B", "BSA", "85", "Presumption as to electronic records and digital signatures",
            "Presumption of integrity of electronic records and digital signatures.",
            "— (rule of evidence)",
            "Renumbered."),
    Mapping("IEA", "114", "BSA", "119", "Court may presume existence of certain facts",
            "Discretionary presumptions (illustrations a–i).",
            "— (rule of evidence)",
            "Renumbered."),
    Mapping("IEA", "114A", "BSA", "120", "Presumption as to absence of consent in certain rape prosecutions",
            "Court shall presume absence of consent in specified aggravated rape categories.",
            "— (rule of evidence)",
            "Renumbered."),
    Mapping("IEA", "—", "BSA", "61", "Admissibility of electronic/digital records — NEW framing",
            "Electronic and digital records to have same legal effect as paper records.",
            "— (rule of evidence)",
            "BSA §61 explicitly states electronic records have same evidentiary value — strengthens digital evidence baseline beyond IEA §65B."),
    Mapping("IEA", "—", "BSA", "23(1) proviso", "Confessions made in immediate presence of Magistrate — NEW clarification",
            "Confession made in immediate presence of Magistrate admissible even if in police custody.",
            "— (rule of evidence)",
            "BSA clarifies the §26 IEA exception more explicitly."),
    Mapping("IEA", "—", "BSA", "23(1)", "Admissibility extended to discovery via electronic records — NEW",
            "Information leading to discovery includes electronic and digital records.",
            "— (rule of evidence)",
            "NEW: discovery doctrine extends to digital evidence (e.g., location pings, deleted messages recovered)."),
]


# ─────────────────────────────────────────────
# Question template generation
# ─────────────────────────────────────────────

QA_TEMPLATES = [
    # Forward lookup: old → new
    ("What is the {new_code} equivalent of {old_code} Section {old_section}?",
     "{old_code} Section {old_section} ({topic}) corresponds to {new_code} Section {new_section}. {description} {change_note}"),

    ("Under the new criminal laws of 2023, which section replaced {old_code} {old_section}?",
     "{new_code} Section {new_section} replaced {old_code} Section {old_section} ({topic}). {description} {change_note}"),

    ("After July 2024, what is the relevant statute for {topic}?",
     "Post-July 2024, the relevant statute for {topic} is {new_code} Section {new_section} (previously {old_code} Section {old_section}). {description}"),

    # Reverse lookup: new → old
    ("What did {new_code} Section {new_section} replace?",
     "{new_code} Section {new_section} ({topic}) replaced {old_code} Section {old_section}. {description} {change_note}"),

    ("Is {new_code} Section {new_section} a new section or a renumbering?",
     "{new_code} Section {new_section} addresses {topic}. {change_note}"),

    # Punishment (for criminal offenses only)
    ("What is the punishment under {new_code} Section {new_section}?",
     "{new_code} Section {new_section} deals with {topic}. Punishment: {punishment}"),

    ("What is the punishment for {topic} under the current Indian criminal law?",
     "Under {new_code} Section {new_section} (the current statute for {topic}), the punishment is: {punishment}. This replaced {old_code} Section {old_section}."),

    # Contextual application
    ("If a case involves {topic} committed in 2025, which section should be cited?",
     "For an offense of {topic} committed after July 1, 2024, the applicable section is {new_code} Section {new_section}. {old_code} Section {old_section} would only apply to offenses committed before that date (per BNSS §531 savings clause)."),

    ("A lawyer drafting a complaint in 2026 alleging {topic} — which statute applies?",
     "The lawyer should cite {new_code} Section {new_section}. {old_code} Section {old_section} has been repealed and applies only to pre-July-2024 offenses. {description}"),

    # Distinction emphasis (catches model errors)
    ("Is {old_code} Section {old_section} still in force?",
     "No. {old_code} Section {old_section} has been repealed effective July 1, 2024. The current corresponding provision is {new_code} Section {new_section} ({topic}). However, {old_code} Section {old_section} continues to apply to offenses committed before July 1, 2024, by virtue of the savings clause in BNSS §531."),
]


# Procedural sections (BNSS) get a slightly different template set — no "punishment".
PROCEDURAL_QA_TEMPLATES = [
    ("What is the {new_code} equivalent of {old_code} Section {old_section}?",
     "{old_code} Section {old_section} ({topic}) corresponds to {new_code} Section {new_section}. {description} {change_note}"),

    ("Under the new criminal procedure code (BNSS), which section replaced {old_code} Section {old_section}?",
     "{new_code} Section {new_section} replaced {old_code} Section {old_section}. It governs {topic}. {change_note}"),

    ("In 2025, what is the procedural provision for {topic}?",
     "The current procedural provision for {topic} is {new_code} Section {new_section}, which replaced {old_code} Section {old_section} on July 1, 2024. {change_note}"),

    ("What did {new_code} Section {new_section} replace?",
     "{new_code} Section {new_section} ({topic}) replaced {old_code} Section {old_section}. {change_note}"),

    ("Has the procedure for {topic} changed under BNSS?",
     "{new_code} Section {new_section} now governs {topic}, replacing {old_code} Section {old_section}. {change_note}"),

    ("If an arrest is made today, which provision governs {topic}?",
     "{new_code} Section {new_section} governs {topic} for any procedure initiated on or after July 1, 2024. {old_code} Section {old_section} would no longer be the operative provision. {change_note}"),
]


# Evidence sections (BSA) — similar structure to procedural.
EVIDENTIARY_QA_TEMPLATES = [
    ("What is the {new_code} equivalent of {old_code} Section {old_section}?",
     "{old_code} Section {old_section} ({topic}) corresponds to {new_code} Section {new_section}. {description} {change_note}"),

    ("Under the Bharatiya Sakshya Adhiniyam, which section replaced {old_code} Section {old_section}?",
     "{new_code} Section {new_section} replaced {old_code} Section {old_section}. It governs {topic}. {change_note}"),

    ("In 2025, which provision of evidence law governs {topic}?",
     "{new_code} Section {new_section} governs {topic} as of July 1, 2024, replacing {old_code} Section {old_section}. {change_note}"),

    ("What did {new_code} Section {new_section} replace?",
     "{new_code} Section {new_section} ({topic}) replaced {old_code} Section {old_section}. {change_note}"),

    ("Is the rule on {topic} different under BSA than under IEA?",
     "{new_code} Section {new_section} now governs {topic}. {change_note} The corresponding earlier provision was {old_code} Section {old_section}."),
]


def is_procedural(mapping: Mapping) -> bool:
    return mapping.new_code == "BNSS"


def is_evidentiary(mapping: Mapping) -> bool:
    return mapping.new_code == "BSA"


def is_new_offense(mapping: Mapping) -> bool:
    """True when this is a brand-new offense with no IPC predecessor (old_section == '—')."""
    return mapping.old_section == "—"


def is_repealed(mapping: Mapping) -> bool:
    """True when an old section has no direct new equivalent (new_section == '—')."""
    return mapping.new_section == "—"


# ─────────────────────────────────────────────
# Sample generation
# ─────────────────────────────────────────────

def generate_samples_for_mapping(m: Mapping) -> list[dict]:
    """
    Generate Q&A pairs for one mapping. Choice of template set depends on category.
    Special cases: new offenses (no IPC predecessor) and repealed sections.
    """
    samples = []

    # Repealed (e.g. IPC 377 for consenting adults) — generate dedicated samples.
    if is_repealed(m):
        samples.append({
            "question": f"What is the {m.new_code} equivalent of {m.old_code} Section {m.old_section}?",
            "answer": (
                f"{m.old_code} Section {m.old_section} ({m.topic}) has NO direct equivalent in "
                f"{m.new_code}. {m.change_note} {m.description}"
            ),
        })
        samples.append({
            "question": f"Is {m.old_code} Section {m.old_section} preserved in the new criminal code?",
            "answer": (
                f"No. {m.old_code} Section {m.old_section} ({m.topic}) was NOT carried over to "
                f"{m.new_code}. {m.change_note}"
            ),
        })
        return samples

    # Entirely new offense (e.g. BNS §111 organised crime) — different question framing.
    if is_new_offense(m):
        samples.append({
            "question": f"What is {m.new_code} Section {m.new_section}?",
            "answer": (
                f"{m.new_code} Section {m.new_section} addresses {m.topic}. {m.description} "
                f"Punishment: {m.punishment}. {m.change_note}"
            ),
        })
        samples.append({
            "question": f"Was {m.topic} an offense under IPC?",
            "answer": (
                f"Not as a standalone offense. {m.new_code} Section {m.new_section} "
                f"({m.topic}) is a new provision introduced in 2023. {m.change_note} "
                f"Punishment: {m.punishment}"
            ),
        })
        samples.append({
            "question": f"If someone is charged with {m.topic} in 2025, what is the statute?",
            "answer": (
                f"The applicable statute is {m.new_code} Section {m.new_section}. "
                f"{m.description} Punishment under {m.new_code} §{m.new_section}: {m.punishment}."
            ),
        })
        return samples

    # Pick template set
    if is_procedural(m):
        templates = PROCEDURAL_QA_TEMPLATES
    elif is_evidentiary(m):
        templates = EVIDENTIARY_QA_TEMPLATES
    else:
        templates = QA_TEMPLATES

    for q_template, a_template in templates:
        q = q_template.format(
            old_code=m.old_code, old_section=m.old_section,
            new_code=m.new_code, new_section=m.new_section,
            topic=m.topic,
        )
        a = a_template.format(
            old_code=m.old_code, old_section=m.old_section,
            new_code=m.new_code, new_section=m.new_section,
            topic=m.topic.lower(), description=m.description,
            punishment=m.punishment, change_note=m.change_note,
        )
        samples.append({"question": q, "answer": a})

    return samples


def generate_dataset() -> list[dict]:
    """Generate the full dataset by iterating over all mappings."""
    all_mappings = BNS_MAPPINGS + BNSS_MAPPINGS + BSA_MAPPINGS
    dataset = []
    for m in all_mappings:
        dataset.extend(generate_samples_for_mapping(m))

    # Shuffle so train/test split mixes BNS/BNSS/BSA evenly
    random.seed(42)
    random.shuffle(dataset)
    return dataset


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bns_bnss_bsa_mapping.jsonl"

    dataset = generate_dataset()

    with open(out_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    n_bns  = len(BNS_MAPPINGS)
    n_bnss = len(BNSS_MAPPINGS)
    n_bsa  = len(BSA_MAPPINGS)

    print(f"Wrote {len(dataset)} samples to {out_path}")
    print(f"Source mappings: BNS={n_bns}, BNSS={n_bnss}, BSA={n_bsa} "
          f"(total {n_bns + n_bnss + n_bsa} seed mappings)")
    print(f"Avg samples per mapping: {len(dataset) / (n_bns + n_bnss + n_bsa):.1f}")
    print("\nFirst 2 samples:")
    for s in dataset[:2]:
        print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
