"""Test Phase 1 + Phase 2 + Phase 3 autonomous solver pipeline."""
import sys, os, io, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

import google.generativeai as genai
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE"))

from parsers.pdf_parser import PDFCrosswordParser
from solvers.autonomous_solver import AutonomousSolver

pdf_path = r"C:\Users\Kenok\OneDrive\Desktop\Crosswords\LATimes.pdf"

print("=== Parsing PDF ===")
parser = PDFCrosswordParser(source=pdf_path)
graph = parser.build_graph()
print(f"Parsed: {graph.width}x{graph.height}, {len(graph.clues)} clues\n")

solver = AutonomousSolver(graph)

print("=== Phase 1: Absolute Certainty Pass ===")
p1 = solver.execute_phase_1_pass()
print(f"Phase 1 locked: {p1}\n")

print("=== Phase 2: Constraint Propagation ===")
p2 = solver.execute_phase_2_pass()
print(f"Phase 2 locked: {p2}\n")

print("=== Phase 3: Local Regex Dictionary ===")
p3 = solver.execute_phase_3_pass()
print(f"Phase 3 locked: {p3}\n")

summary = solver.get_solve_summary()
print(f"=== Final Summary ===")
print(f"Phase 1: {p1} | Phase 2: +{p2} | Phase 3: +{p3}")
print(f"Total: {summary['solved_clues']} / {summary['total_clues']} ({summary['solve_percentage']}%)")
print(f"Grid cells filled: {summary['filled_cells']} / {summary['total_cells']}")
