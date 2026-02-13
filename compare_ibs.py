"""
IBS Similarity Analysis for 23andMe Genotyping Data
=====================================================

This script performs comprehensive Identity By State (IBS) analysis on 23andMe
genotyping data files. It computes pairwise similarity scores between all 
individuals based on autosomal SNPs using rsID matching across different chip versions.

Key Features:
  • Parses 23andMe text files and extracts autosomal genotypes
  • Computes IBS similarity scores using multiset intersection of alleles
  • Generates:
    - Pairwise comparison table (console output)
    - Similarity matrix heatmap (PNG image)
    - Hierarchical clustering dendrogram with distance labels (PNG image)
"""

import sys
import re
import glob
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# ============================================================================
# Constants
# ============================================================================
AUTOSOMES = set(str(i) for i in range(1, 23))
GENOTYPES = re.compile(r'^[ACGT]{2}$', re.IGNORECASE)


def parse_23andme(path):
    """
    Parse a 23andMe genotyping file and extract autosomal SNPs.
    
    Only processes:
      • Autosomal chromosomes (1-22), excluding X, Y, MT
      • Valid 2-letter ACGT genotypes (both alleles called)
      • SNPs with rsID matching pattern 'rs*'
    
    Args:
        File path to 23andMe text file
        
    Returns:
        Dictionary mapping rsID -> genotype (uppercase)
    """
    dic = {}
    with open(path, 'r', encoding='utf-8') as fileread:
        for line in fileread:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            rsid, chrom, pos, geno = parts[0], parts[1], parts[2], parts[3]
            # require an rsID to use as the key
            if not rsid or not rsid.lower().startswith('rs'):
                continue
            # normalize chromosome (tolerate "chr" prefix) and keep only autosomes
            chrom_norm = chrom.lower().replace('chr', '')
            if chrom_norm not in AUTOSOMES:
                continue
            if not pos.isdigit():
                continue
            geno = geno.upper()
            # skip uncalled / partially called genotypes; accept only two A/C/G/T letters
            if not GENOTYPES.match(geno):
                continue
            key = rsid  # use rsID rather than (chrom,pos) to handle different builds/chip versions
            dic[key] = geno
    return dic


def ibs_score(geno1, geno2):
    """
    Compute Identity By State (IBS) score between two genotypes.
    
    Uses multiset intersection to count shared allele copies:
      • 0 = no shared alleles
      • 1 = one shared allele
      • 2 = both alleles shared
    
    Args:
        First and second genotype as 2-letter string (e.g., 'AA', 'AT', 'GC')
        
    Returns:
        Integer IBS score (0, 1, or 2)
    """
    count1 = Counter(list(geno1))
    count2 = Counter(list(geno2))
    # sum of min counts across alleles gives number of shared allele copies (0..2)
    shared = sum(min(count1[a], count2[a]) for a in set(count1.keys()) | set(count2.keys()))
    # Ensure value is 0,1,2
    return max(0, min(2, shared))


def compare_files(path1, path2):
    """
    Compare two 23andMe files and compute IBS similarity score.
    
    Parses both files, identifies overlapping autosomal SNPs, computes
    IBS scores at each locus, and returns normalized similarity metric.
    
    Args:
        Path to first and second 23andMe files
        
    Returns:
        Dictionary with comparison results or None if no overlap found:
          • 'file1': Path to first file
          • 'file2': Path to second file
          • 'n_loci': Number of overlapping SNPs
          • 'total_score': Sum of IBS scores (0 to 2*n_loci)
          • 'max_score': Maximum possible score (2*n_loci)
          • 'average_similarity': Normalized similarity (0 to 1)
    """
    dic1 = parse_23andme(path1)
    dic2 = parse_23andme(path2)

    common_keys = set(dic1.keys()) & set(dic2.keys())
    if not common_keys:
        print("No overlapping autosomal SNPs found between the two files.", file=sys.stderr)
        return None

    total_score = 0
    for key in common_keys:
        total_score += ibs_score(dic1[key], dic2[key])

    n_loci = len(common_keys)
    max_score = 2 * n_loci
    avg = total_score / max_score

    return {
        'file1': path1,
        'file2': path2,
        'n_loci': n_loci,
        'total_score': total_score,
        'max_score': max_score,
        'average_similarity': avg,
    }


def create_dendrogram(similarity_matrix, names):
    """
    Create a hierarchical clustering dendrogram from similarity matrix.
    Converts similarity to distance (1 - similarity) and uses Ward linkage.
    Adds distance values as labels on branches.

    Args:
        similarity_matrix (np.ndarray): Square matrix of pairwise similarity
            values as percentage.
        names (list[str]): Ordered list of individual names corresponding to
            rows/columns of `similarity_matrix`.
    """
    # Convert similarity to distance (most similar = closest)
    distance_matrix = 1 - (similarity_matrix / 100.0)
    
    # Convert square distance matrix to condensed form for linkage
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering using Ward method with Euclidean metric
    Z = linkage(condensed_dist, method='ward', metric='euclidean')
    
    # Create dendrogram with distance annotations
    fig, ax = plt.subplots(figsize=(16, 9))
    dendro = dendrogram(Z, labels=names, leaf_font_size=11, leaf_rotation=45, ax=ax)
    
    # Add distance labels on branches
    for i, d in zip(dendro['icoord'], dendro['dcoord']):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        ax.text(x, y, f'{y:.3f}', va='center', ha='center', fontsize=15,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))
    
    ax.set_title('Hierarchical Clustering Dendrogram (IBS Similarity)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Individual', fontsize=13, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('ibs_dendrogram.png', dpi=300, bbox_inches='tight')
    print("Dendrogram saved as 'ibs_dendrogram.png'")
    plt.close()


def get_individual_name(filepath):
    """
    Extract an individual's name from a 23andMe filename.

    Args:
        Path or filename of the 23andMe file.

    Returns:
        Extracted individual name.
    """
    basename = os.path.basename(filepath)
    if "DNA23andMe" in basename:
        return basename.split("DNA23andMe")[0]
    return basename.replace(".txt", "")


def main():
    """Main function to analyze IBS similarity across all 23andMe files."""
    # ========================================================================
    # Step 1: Collect and validate input files
    # ========================================================================
    txt_files = sorted(glob.glob('*.txt'))
    
    if not txt_files:
        print("Error: No .txt files found in the current directory.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"IBS Similarity Analysis")
    print(f"{'='*80}")
    print(f"Found {len(txt_files)} 23andMe files:")
    for f in txt_files:
        print(f"  • {f}")
    print()
    
    # ========================================================================
    # Step 2: Extract individual names and sort them
    # ========================================================================
    names = [get_individual_name(f) for f in txt_files]
    name_to_file = {name: file for name, file in zip(names, txt_files)}
    
    # Sort names alphabetically for consistent matrix ordering
    sorted_names = sorted(names)
    sorted_files = [name_to_file[name] for name in sorted_names]
    
    # ========================================================================
    # Step 3: Build similarity matrix using sorted order
    # ========================================================================
    n_individuals = len(sorted_files)
    similarity_matrix = np.zeros((n_individuals, n_individuals))
    results = []
    
    print("Computing pairwise IBS similarities...")
    for i, file1 in enumerate(sorted_files):
        for j, file2 in enumerate(sorted_files):
            if i <= j:
                result = compare_files(file1, file2)
                if result is not None:
                    similarity = result['average_similarity'] * 100
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # Symmetric matrix
                    
                    if i <= j:  # Only add upper triangle to results table
                        results.append({
                            'Individual 1': sorted_names[i],
                            'Individual 2': sorted_names[j],
                            'Loci': result['n_loci'],
                            'Total Score': result['total_score'],
                            'Similarity': f"{similarity:.2f}%"
                        })
    
    print("Similarity computation complete.\n")
    
    # ========================================================================
    # Step 4: Display results table
    # ========================================================================
    if results:
        print("=" * 110)
        print(f"{'Individual 1':<25} {'Individual 2':<25} {'Loci':<10} {'Total Score':<15} {'Similarity':<15}")
        print("=" * 110)
        for row in results:
            print(f"{row['Individual 1']:<25} {row['Individual 2']:<25} {row['Loci']:<10} {row['Total Score']:<15} {row['Similarity']:<15}")
        print("=" * 110 + "\n")
        
        # ====================================================================
        # Step 5: Generate similarity matrix heatmap
        # ====================================================================
        print("Generating similarity matrix heatmap...")
        fig, ax = plt.subplots(figsize=(13, 11))
        # Flip the matrix vertically so the y-axis shows names in inverted order
        heatmap_data = similarity_matrix[::-1, :]
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=sorted_names, yticklabels=list(reversed(sorted_names)), 
                cbar_kws={'label': 'Similarity (%)', 'shrink': 0.8},
                vmin=0, vmax=100, square=True, linewidths=0.5, 
                cbar=True, ax=ax)
        ax.set_title('IBS Similarity Comparison Matrix', 
                 fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Individual', fontsize=12, fontweight='bold')
        ax.set_ylabel('Individual', fontsize=12, fontweight='bold')
        # Keep y-axis labels horizontal for readability
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        plt.savefig('ibs_similarity_grid.png', dpi=300, bbox_inches='tight')
        print("Grid comparison saved as 'ibs_similarity_grid.png'\n")
        plt.close()
        
        # ====================================================================
        # Step 6: Generate hierarchical clustering dendrogram
        # ====================================================================
        print("Generating hierarchical clustering dendrogram...")
        create_dendrogram(similarity_matrix, sorted_names)
        
        print(f"\n{'='*80}")
        print("Analysis complete!")
        print(f"{'='*80}\n")
    else:
        print("Error: No valid comparisons could be made.", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()