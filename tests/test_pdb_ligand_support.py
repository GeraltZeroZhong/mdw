from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


ALANINE_DIPEPTIDE_PDB = """\
REMARK   1 CREATED WITH OPENMM 8.2, 2024-12-03
HETATM    1  CH3 ACE A   1       2.000   2.090   0.000  1.00  0.00           C
HETATM    2  C   ACE A   1       3.427   2.641  -0.000  1.00  0.00           C
HETATM    3  O   ACE A   1       4.391   1.877  -0.000  1.00  0.00           O
ATOM      4  N   ALA A   2       3.555   3.970  -0.000  1.00  0.00           N
ATOM      5  CA  ALA A   2       4.853   4.614  -0.000  1.00  0.00           C
ATOM      6  CB  ALA A   2       5.661   4.221  -1.232  1.00  0.00           C
ATOM      7  C   ALA A   2       4.713   6.129   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       3.601   6.653   0.000  1.00  0.00           O
HETATM    9  N   NME A   3       5.846   6.835   0.000  1.00  0.00           N
HETATM   10  C   NME A   3       5.846   8.284   0.000  1.00  0.00           C
TER      11      NME A   3
CONECT    1    2
CONECT    2    1    3    4
CONECT    3    2
CONECT    4    2
CONECT    7    9
CONECT    9    7   10
CONECT   10    9
END
"""


def _translate_pdb(text: str, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38]) + dx
            y = float(line[38:46]) + dy
            z = float(line[46:54]) + dz
            line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _write_pdb(path: Path, text: str = ALANINE_DIPEPTIDE_PDB) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_pdb_ligand_config(root: Path):
    from md_workbench.config import WorkflowConfig

    receptor = _write_pdb(root / "inputs" / "receptor.pdb")
    ligand = _write_pdb(root / "inputs" / "ligand_input.pdb", _translate_pdb(ALANINE_DIPEPTIDE_PDB, dx=20.0))

    cfg = WorkflowConfig(workspace_root=str(root))
    cfg.prep.preprocess_mode = "never"
    cfg.prep.receptor_input = str(receptor)
    cfg.prep.receptor_output = str(root / "work" / "prep" / "prepared_receptor.pdb")
    cfg.docking.ligand_input_mode = "pdb"
    cfg.docking.docking_mode = "skip"
    cfg.docking.ligand_pdb_input = str(ligand)
    cfg.docking.ligand_output_pdb = str(root / "work" / "prep" / "prepared_ligand.pdb")
    cfg.docking.ligand_output_sdf = str(root / "work" / "prep" / "prepared_ligand.sdf")
    cfg.docking.extracted_pose_sdf = str(root / "work" / "docking" / "best_ligand.sdf")
    cfg.run.output_root = str(root / "work" / "md")
    cfg.run.protein_pdb = ""
    cfg.run.ligand_sdf = ""
    return cfg


class PdbLigandConfigTests(unittest.TestCase):
    def test_preflight_accepts_pdb_ligand_with_skip_docking(self):
        from md_workbench.core.validation import preflight_validate

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_pdb_ligand_config(Path(tmp))
            result = preflight_validate(cfg)

        self.assertEqual([], result.errors)
        self.assertTrue(result.ok)
        self.assertTrue(any("skip" in warning for warning in result.warnings))

    def test_run_input_path_inference_uses_prepared_or_input_pdb(self):
        from md_workbench.core.pathing import infer_run_input_paths

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_pdb_ligand_config(Path(tmp))
            protein, ligand = infer_run_input_paths(cfg)
            self.assertEqual(cfg.prep.receptor_output, protein)
            self.assertEqual(cfg.docking.ligand_output_pdb, ligand)

            cfg.do_prep = False
            protein, ligand = infer_run_input_paths(cfg)
            self.assertEqual(cfg.prep.receptor_input, protein)
            self.assertEqual(cfg.docking.ligand_pdb_input, ligand)

    def test_pdb_ligand_prep_skip_copies_inputs_without_docking(self):
        try:
            from md_workbench.prep.workflow import run_prep_workflow
        except Exception as exc:  # pragma: no cover - dependency-dependent skip
            self.skipTest(f"prep workflow dependencies are unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_pdb_ligand_config(Path(tmp))
            outputs = run_prep_workflow(cfg.prep, cfg.docking)

            self.assertEqual(cfg.prep.receptor_output, outputs["receptor"])
            self.assertEqual(cfg.docking.ligand_output_pdb, outputs["ligand"])
            self.assertEqual({"skipped": True}, outputs["docking"])
            self.assertTrue(Path(cfg.prep.receptor_output).is_file())
            self.assertTrue(Path(cfg.docking.ligand_output_pdb).is_file())
            self.assertEqual(Path(cfg.docking.ligand_pdb_input).read_text(), Path(cfg.docking.ligand_output_pdb).read_text())

    def test_pdb_ligand_rejects_non_skip_docking_before_writing_outputs(self):
        try:
            from md_workbench.prep.workflow import run_prep_workflow
        except Exception as exc:  # pragma: no cover - dependency-dependent skip
            self.skipTest(f"prep workflow dependencies are unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_pdb_ligand_config(Path(tmp))
            cfg.docking.docking_mode = "auto"

            with self.assertRaisesRegex(ValueError, "docking_mode='skip'"):
                run_prep_workflow(cfg.prep, cfg.docking)

            self.assertFalse(Path(cfg.prep.receptor_output).exists())
            self.assertFalse(Path(cfg.docking.ligand_output_pdb).exists())

    def test_preflight_rejects_pdb_ligand_auto_docking_without_binary_noise(self):
        from md_workbench.core.validation import preflight_validate

        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_pdb_ligand_config(Path(tmp))
            cfg.docking.docking_mode = "auto"
            result = preflight_validate(cfg)

        self.assertFalse(result.ok)
        self.assertTrue(any("docking_mode='skip'" in error for error in result.errors))
        self.assertFalse(any("Required docking binary" in error for error in result.errors))

    def test_non_sdf_ligand_torsion_detection_is_empty(self):
        try:
            from md_workbench.analysis.basic.extended import detect_ligand_rotatable_dihedrals
        except Exception as exc:  # pragma: no cover - dependency-dependent skip
            self.skipTest(f"analysis dependencies are unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            ligand = _write_pdb(Path(tmp) / "ligand.pdb")
            self.assertEqual([], detect_ligand_rotatable_dihedrals(str(ligand), [0, 1, 2]))


class PdbPeptideSystemBuilderTests(unittest.TestCase):
    def test_pdb_peptide_ligand_builds_forcefield_system_and_tags_output_residues(self):
        try:
            import mdtraj as md
            from openmm.app import HBonds, NoCutoff, PDBFile
            from md_workbench.core import find_ligand_residues, ligand_residue_summary
            from md_workbench.simulate.system_builder import build_modeller_and_forcefield, tag_ligand_residues_for_output
        except Exception as exc:  # pragma: no cover - dependency-dependent skip
            self.skipTest(f"OpenMM/MDTraj stack is unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receptor = _write_pdb(root / "receptor.pdb")
            ligand = _write_pdb(root / "ligand.pdb", _translate_pdb(ALANINE_DIPEPTIDE_PDB, dx=20.0))

            modeller, forcefield, ligand_info = build_modeller_and_forcefield(str(receptor), str(ligand))
            system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

            self.assertEqual("pdb", ligand_info.input_format)
            self.assertEqual((1,), ligand_info.ligand_chain_indices)
            self.assertEqual(system.getNumParticles(), modeller.topology.getNumAtoms())

            tag_ligand_residues_for_output(modeller.topology, ligand_info)
            tagged_path = root / "tagged_complex.pdb"
            with tagged_path.open("w", encoding="utf-8") as handle:
                PDBFile.writeFile(modeller.topology, modeller.positions, handle)

            tagged = md.load_pdb(str(tagged_path))
            ligand_residues = find_ligand_residues(tagged.topology)
            self.assertEqual(3, len(ligand_residues))
            self.assertTrue(all(residue.name == "LIG" for residue in ligand_residues))
            self.assertIn("3 residues", ligand_residue_summary(ligand_residues))


if __name__ == "__main__":
    unittest.main()
