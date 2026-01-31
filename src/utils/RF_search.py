"""
Hyperparameter search for ResNet-18 b1 on CIFAR-100
Find optimal configuration for recommended RF range
"""

import itertools
import pandas as pd

# Considered w.r.t the RF size before NECK 
def calc_rf(k1, s1, p1, use_pool, k2=3, s2=2, p2=1):
    rf, stride_prod, size = 1, 1, 32
    
    # b1: conv1
    rf += (k1 - 1) * stride_prod
    stride_prod *= s1
    size = (size + 2*p1 - k1) // s1 + 1
    
    # b1: maxpool
    if use_pool:
        rf += (k2 - 1) * stride_prod
        stride_prod *= s2
        size = (size + 2*p2 - k2) // s2 + 1

    b1_ratio = rf/32
    
    # layer2: 4×(3×3, s=1)
    rf += 4 * (3 - 1) * stride_prod
    
    # layer3...5: (3×3, s=2) + 3×(3×3, s=1)
    for _ in range(3):
        rf += (3 - 1) * stride_prod
        stride_prod *= 2
        size = (size + 2 - 3) // 2 + 1
        rf += 3 * (3 - 1) * stride_prod
    
    return rf, size, b1_ratio


def search():
    # Target: Let's say that CIFAR-100 objects occupy ~75% of image (24/32 pixels)
    # that means that we can limit the RF just before the layer to a range that 
    # should encompass (due to Global Avg Pooling) the entire image or more, with the minimum 
    # estimated amount being 75 (and let's say maximum 5x the image size)

    # Spatial resolutions any lower than 2x2 aren't considered useful
    
    
    rf_min, rf_max = 24, 5*32
    min_spatial = 2
    
    results = []
    
    for k1, s1, pool in itertools.product([3, 5,7], [1, 2], [False, True]):
        p1 = (k1 - 1) // 2
        rf, spatial, b1_ratio = calc_rf(k1, s1, p1, pool)
        
        rf_ok = rf_min <= rf <= rf_max
        spatial_ok = spatial >= min_spatial
        status = "GOOD" if (rf_ok and spatial_ok) else ("PARTIAL" if (rf_ok or spatial_ok) else "BAD")
        
        results.append({
            'k1': k1, 's1': s1, 'p1': p1, 'pool': pool,
            'RF': rf, 'spatial': spatial,
            'RF_ok': rf_ok, 'spatial_ok': spatial_ok, 'status': status,
            'b1_ratio': b1_ratio
        })
    
    df = pd.DataFrame(results)

    # L1 metric
    mean_rf = (rf_min + rf_max)/2
    df['rf_dist'] = abs(df['RF'] - mean_rf)
    df = df.sort_values(['status', 'rf_dist'])
    
    print("="*70)
    print(f"TARGET: RF in [{rf_min}, {rf_max}]px, spatial >= {min_spatial}x{min_spatial}")
    print("="*70)
    
    for _, row in df.iterrows():
        pool_str = "pool" if row['pool'] else "no_pool"
        print(f"k={row['k1']} s={row['s1']} p={row['p1']} {pool_str:8s} -> "
              f"RF={row['RF']:3d}px ({row['spatial']}x{row['spatial']}) (b1_ratio = {row['b1_ratio']}) [{row['status']}]")
    
    print("\n" + "="*70)
    print("BEST CONFIG:")
    print("="*70)
    
    good = df[df['status'] == 'GOOD']
    if len(good) > 0:
        best = good.iloc[0]
        print(f"kernel={best['k1']}, stride={best['s1']}, padding={best['p1']}, "
              f"maxpool={best['pool']}")
        print(f"Final RF: {best['RF']}px, Spatial: {best['spatial']}x{best['spatial']}")
    else:
        print("No configuration meets all criteria. Best partial:")
        best = df[df['status'] == 'PARTIAL'].iloc[0]
        print(f"kernel={best['k1']}, stride={best['s1']}, padding={best['p1']}, "
              f"maxpool={best['pool']}")
        print(f"Final RF: {best['RF']}px, Spatial: {best['spatial']}x{best['spatial']}")
    
    print("="*70)
    
    print(f"Best receptive field ratio (after just b1)= {best['b1_ratio']*100:.2f} %")

    print("="*70)


if __name__ == "__main__":
    search()