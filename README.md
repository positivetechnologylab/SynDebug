# SynDebug

Quantum program assertions for quantum computers using QEC codes.

The repo is organized as follows:

patch_register.py contains the code for QecProgram, PatchRegister, and CodePatch classes.

logical_gadgets.py contains code for transversal logical operators used in various demos.

qec_assertions.py contains the assertion method definitions.

test_init_assert.py and test_entangle_assert.py contain the code for generating the data in table 2 and table 3 respectively

test_init_assert_exhaustive.py and test_entangle_assert.py exhaustively iterate through combinations of code type, distance, and initializations.

Saved versions of the paper results are stored in the results folder