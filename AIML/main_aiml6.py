#!/usr/bin/env python3
import sys
from typing import Callable, Dict, List, Optional, Set, Tuple

class WorkingMemory:
    def __init__(self) -> None:
        self.facts: Set[str] = set()
        self.history: List[Tuple[str, str]] = []

    def assert_fact(self, fact: str, source: str) -> bool:
        if fact in self.facts:
            return False
        self.facts.add(fact)
        self.history.append((fact, source))
        return True

    def holds(self, fact: str) -> bool:
        return fact in self.facts

class Rule:
    def __init__(self, name: str, conditions: List[str], conclusion: str) -> None:
        self.name: str = name
        self.conditions: List[str] = conditions
        self.conclusion: str = conclusion

    def applicable(self, memory: WorkingMemory) -> bool:
        for cond in self.conditions:
            if cond.startswith("not "):
                if memory.holds(cond[4:]):
                    return False
            elif not memory.holds(cond):
                return False
        return True

class InferenceEngine:
    def __init__(self, rules: List[Rule], memory: WorkingMemory) -> None:
        self.rules: List[Rule] = rules
        self.memory: WorkingMemory = memory

    def run(self) -> List[Rule]:
        fired: List[Rule] = []
        done: Set[str] = set()
        changed: bool = True
        while changed:
            changed = False
            for rule in self.rules:
                if rule.name in done or not rule.applicable(self.memory):
                    continue
                done.add(rule.name)
                if self.memory.assert_fact(rule.conclusion, rule.name):
                    fired.append(rule)
                    changed = True
        return fired

QUESTIONS: List[Tuple[str, str, Optional[str]]] = [
    ("power_led_off", "Is the power LED off when you press the power button?", None),
    ("outlet_works", "Does the wall outlet work with other devices?", "power_led_off"),
    ("fans_not_spinning", "Are all case fans completely still?", "power_led_off"),
    ("no_display", "Is the screen blank after boot?", None),
    ("fans_spin", "Do the fans spin up at power-on?", "no_display"),
    ("beeps_on_boot", "Does the PC emit beep codes at boot?", "no_display"),
    ("monitor_works_elsewhere", "Does the monitor work with another device?", "no_display"),
    ("slow", "Is the PC running slowly?", None),
    ("high_cpu_temp", "Is the CPU temperature above 85 C under load?", None),
    ("disk_almost_full", "Is the system drive more than 90% full?", None),
    ("hdd_drive", "Is the system drive a mechanical hard disk?", None),
    ("high_disk_usage", "Does disk usage sit near 100% in Task Manager?", None),
    ("random_restarts", "Does the PC restart or shut down randomly?", None),
    ("new_hardware_added", "Was new hardware added recently?", None),
    ("clicking_noise", "Does the drive make clicking or grinding sounds?", None),
    ("no_internet", "Is the internet unavailable on this PC?", None),
    ("other_devices_online", "Do other devices on the network have internet?", "no_internet"),
]

RULES: List[Rule] = [
    Rule("R01", ["power_led_off", "outlet_works"], "psu_suspect"),
    Rule("R02", ["psu_suspect", "fans_not_spinning"], "dx_psu_failure"),
    Rule("R03", ["no_display", "beeps_on_boot"], "ram_suspect"),
    Rule("R04", ["ram_suspect"], "dx_ram_fault"),
    Rule("R05", ["no_display", "fans_spin", "not beeps_on_boot"], "video_suspect"),
    Rule("R06", ["video_suspect", "monitor_works_elsewhere"], "dx_gpu_fault"),
    Rule("R07", ["slow", "high_cpu_temp"], "thermal_throttling"),
    Rule("R08", ["thermal_throttling"], "dx_cooling"),
    Rule("R09", ["random_restarts", "high_cpu_temp"], "dx_overheat_shutdown"),
    Rule("R10", ["random_restarts", "new_hardware_added"], "dx_psu_underpowered"),
    Rule("R11", ["slow", "disk_almost_full"], "dx_disk_full"),
    Rule("R12", ["slow", "hdd_drive", "high_disk_usage"], "dx_slow_hdd"),
    Rule("R13", ["clicking_noise"], "dx_drive_failing"),
    Rule("R14", ["no_internet", "other_devices_online"], "dx_local_network"),
    Rule("R15", ["no_internet", "not other_devices_online"], "dx_router_isp"),
]

DIAGNOSES: Dict[str, Tuple[str, str]] = {
    "dx_psu_failure": ("Power supply failure", "Test with a known-good PSU."),
    "dx_ram_fault": ("RAM fault", "Reseat the modules and test one stick at a time."),
    "dx_gpu_fault": ("Graphics card fault", "Reseat the GPU or test with integrated graphics."),
    "dx_cooling": ("Thermal throttling", "Clean the fans and heatsink, then reapply thermal paste."),
    "dx_overheat_shutdown": ("Overheating shutdowns", "Fix cooling and verify fan curves in BIOS."),
    "dx_psu_underpowered": ("Underpowered PSU", "Check total wattage against the new hardware."),
    "dx_disk_full": ("System drive nearly full", "Free at least 15% of the drive."),
    "dx_slow_hdd": ("Slow mechanical drive", "Upgrade the system drive to an SSD."),
    "dx_drive_failing": ("Drive failing", "Back up data immediately and replace the drive."),
    "dx_local_network": ("Local network fault", "Reset the network adapter and renew the IP lease."),
    "dx_router_isp": ("Router or ISP outage", "Restart the router and contact the ISP."),
}

DEMO_ANSWERS: Dict[str, bool] = {
    "slow": True,
    "high_cpu_temp": True,
    "disk_almost_full": True,
    "hdd_drive": True,
    "high_disk_usage": True,
    "random_restarts": True,
}

def collect(ask_fn: Callable[[str, str], bool], memory: Optional[WorkingMemory] = None) -> WorkingMemory:
    memory = memory or WorkingMemory()
    for fact, text, requires in QUESTIONS:
        if requires and not memory.holds(requires):
            continue
        if ask_fn(fact, text):
            memory.assert_fact(fact, "user")
    return memory

def format_report(memory: WorkingMemory) -> List[str]:
    found = [f for f, _ in memory.history if f in DIAGNOSES]
    if not found:
        return ["No diagnosis could be reached from the given symptoms."]
    lines: List[str] = []
    for number, fact in enumerate(found, 1):
        title, advice = DIAGNOSES[fact]
        lines.append(f"{number}. {title}: {advice}")
    return lines

def diagnose(ask_fn: Callable[[str, str], bool]) -> Tuple[WorkingMemory, List[Rule]]:
    memory = collect(ask_fn)
    fired = InferenceEngine(RULES, memory).run()
    return memory, fired

def interactive_ask(fact: str, text: str) -> bool:
    while True:
        try:
            answer = input(f"{text} (y/n): ").strip().lower()
        except EOFError:
            return False
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y or n.")

def main() -> None:
    if "--demo" in sys.argv:
        ask: Callable[[str, str], bool] = lambda fact, text: DEMO_ANSWERS.get(fact, False)
    else:
        print("PC Diagnostic Expert System")
        ask = interactive_ask
    memory, fired = diagnose(ask)
    
    print("\nRules fired:")
    for rule in fired:
        print(f"  {rule.name}: {' & '.join(rule.conditions)} -> {rule.conclusion}")
        
    print("\nDiagnosis:")
    for line in format_report(memory):
        print(f"  {line}")

if __name__ == "__main__":
    main()