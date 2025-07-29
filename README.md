# ai-security-labs

Hands-on labs for AI/ML/LLM Security. Simulating OWASP ML Top 10 attacks and defenses, including adversarial input manipulation, model theft, and prompt injection.

## Labs

- ✅ ML01: Input Manipulation (FGSM attack + defense)
- 🛠️ Coming soon: Data poisoning, model inversion, prompt injection...

## Usage

```bash
pip install -r requirements.txt
cd owasp/ML01_input_manipulation
python owasp/ML01_input_manipulation/attack_fgsm.py     # Run attack
python owasp/ML01_input_manipulation/defense_fgsm.py    # Run detection & defense

## License
See the [LICENSE](./LICENSE) file for full details.
