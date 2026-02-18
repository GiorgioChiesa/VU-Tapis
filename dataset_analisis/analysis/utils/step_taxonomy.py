"""
Step and Phase Taxonomy for GraSP dataset.
Provides metadata and classification for surgical steps and phases.
"""

from config.analysis_config import (
    STEP_CATEGORIES,
    PHASE_CATEGORIES,
    STEP_TAXONOMY,
    get_step_name,
    get_phase_name,
    get_step_class,
)


class StepTaxonomy:
    """Manages step metadata and classification."""

    def __init__(self):
        self.step_names = STEP_CATEGORIES
        self.phase_names = PHASE_CATEGORIES
        self.step_taxonomy = STEP_TAXONOMY

    def classify_step(self, step_id: int) -> str:
        """
        Classify a step into a taxonomy category.

        Args:
            step_id: Step ID (0-20)

        Returns:
            Category name (e.g. 'dissection', 'idle', 'suturing')
        """
        return get_step_class(step_id)

    def get_steps_by_category(self, category: str) -> list:
        """
        Get all step IDs for a given taxonomy category.

        Args:
            category: Category name

        Returns:
            List of step IDs
        """
        return self.step_taxonomy.get(category, [])

    def is_procedural_step(self, step_id: int) -> bool:
        """Check if a step is procedural (not idle)."""
        return step_id != 0

    def is_idle(self, step_id: int) -> bool:
        """Check if a step is idle."""
        return step_id == 0

    def get_step_metadata(self, step_id: int) -> dict:
        """
        Get comprehensive metadata for a step.

        Args:
            step_id: Step ID

        Returns:
            Dictionary with step metadata
        """
        return {
            'id': step_id,
            'name': self.step_names.get(step_id, 'Unknown'),
            'category': self.classify_step(step_id),
            'is_idle': self.is_idle(step_id),
            'is_procedural': self.is_procedural_step(step_id),
        }

    def get_phase_metadata(self, phase_id: int) -> dict:
        """Get comprehensive metadata for a phase."""
        return {
            'id': phase_id,
            'name': self.phase_names.get(phase_id, 'Unknown'),
            'is_idle': phase_id == 0,
        }

    def get_all_steps_metadata(self) -> list:
        """Get metadata for all steps."""
        return [self.get_step_metadata(sid) for sid in sorted(self.step_names.keys())]

    def get_all_phases_metadata(self) -> list:
        """Get metadata for all phases."""
        return [self.get_phase_metadata(pid) for pid in sorted(self.phase_names.keys())]


class PhaseTaxonomy:
    """Manages phase metadata."""

    def __init__(self):
        self.phase_names = PHASE_CATEGORIES

    def get_phase_name(self, phase_id: int) -> str:
        """Get phase name by ID."""
        return self.phase_names.get(phase_id, f'Unknown_Phase_{phase_id}')

    def is_idle_phase(self, phase_id: int) -> bool:
        """Check if phase is idle."""
        return phase_id == 0


if __name__ == '__main__':
    # Test
    step_tax = StepTaxonomy()
    print("Step Taxonomy Test:")
    print(f"Total steps: {len(step_tax.step_names)}")
    print(f"Step 0: {step_tax.step_names[0]} -> {step_tax.classify_step(0)}")
    print(f"Step 1: {step_tax.step_names[1]} -> {step_tax.classify_step(1)}")
    print(f"Dissection steps: {step_tax.get_steps_by_category('dissection')}")
