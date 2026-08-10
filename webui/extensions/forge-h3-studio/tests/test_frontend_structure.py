from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendStructureTests(unittest.TestCase):
    def test_modal_actions_do_not_use_host_hidden_footer_element(self):
        script = (ROOT / "javascript" / "h3_studio.js").read_text(encoding="utf-8")
        css = (ROOT / "style.css").read_text(encoding="utf-8")

        self.assertNotIn("<footer>", script)
        self.assertIn('class="h3s-crop-footer"', script)
        self.assertIn('data-role="crop-zoom"', script)
        self.assertIn('data-action="apply-crop"', script)
        self.assertIn('class="h3s-job-detail-footer"', script)
        self.assertRegex(
            css,
            re.compile(
                r"\.h3s-crop-modal\s*>\s*\.h3s-crop-footer\s*\{[^}]*"
                r"display:\s*flex\s*!important",
                re.DOTALL,
            ),
        )


if __name__ == "__main__":
    unittest.main()
