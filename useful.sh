#!/bin/bash

# Array of network names
networks=("Vis" "SomMot" "DorsAttn" "SalVentAttn" "Limbic" "Cont" "Default" "Hippocampus" "Thalamus" "Cerebellum-Cortex")

# Subjects from sub01 to sub15
for network in "${networks[@]}"; do
  for i in {1..15}; do
    subject=$(printf "sub%02d" $i)

    copy_src="./$network/rdm_lobo_vect_new/$subject"
    src="./$network/rdm_blocks_vect_new/$subject"
    dest="./$network/rdm_lobotomized/"

    # Create destination folder
    mkdir -p "$dest"

    # Sync files matching pattern
    rsync -av --ignore-existing "$copy_src" "$dest"/
    rsync -av "$src"/rand-0*.npy "$dest"/"$subject"/
  done
done

# Subjects from sub01 to sub15
for i in {1..15}; do
  subject=$(printf "sub%02d" $i)
  
  src="./rdm_blocks_new/$subject"
  dest="./rdm_lobotomized/$subject"

  # Create destination folder
  mkdir -p "$dest"

  # Sync files matching pattern
  rsync -av "$src"/rand-0*.npy "$dest"/
done