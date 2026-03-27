# Transition Probability Estimation

This bit of code was written during a hackathon between 2026-03-23 and
2026-03-27 (and a bit later) by Juee Dhar and Pranav Minasandra. The code deals
with (a) pooling together existing data and metadata about baboon sleep, (b)
estimating unbiased transition rates under the Markov assumption between sleep
and wake states, and (c) runs simulations to prove that the debiasing works well.

----------------------------------------------------------

## Setup

1. Choose any folder as your 'PROJECTROOT', and note its path.

2. Create the following subfolders:

    ```
    PROJECTROOT
    |
    |--code/
    |--Data/
    |--Figures/
    
    ```

    **Note**: folder names are case sensitive.

3. In the `Data/` subfolder, accumulate the following files from the EAS data
   server:

   ```
   baboon_sleep_wake_transitions.parquet        cluster_labels.csv           individual_night_locations.csv      metadata.csv      populate_mastersheet.py
    Baboons-MBRP-Mpala-Kenya-reference-data.csv  combined_sleep_analysis.csv  individual_night_locations.parquet  metadata.parquet

   ```

   (Juee is working on automating this)

4. Moving to the 'PROJECTROOT' directory, run `git clone
   https://github.com/jueedhar/sleep-transitions code/`

5. Add a single file inside the `code/` folder called `.cw`. The contents of
   this file should be the text path to the PROJECTROOT.

6. Enter the code directory. To run analyses, enter the command

    ```bash
    python main.py
    ```

    And to run simulations, enter the command

    ```bash
    python runsims.py
    ```

    (depending on your setup, you might need to run `python3` instead of
    `python`.

-------------------------------------------------------------------

## Unbiased relation estimation

(coming soon)

## Sleep transitions manuscript

(coming soon)
