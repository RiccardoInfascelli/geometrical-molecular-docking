#ifndef CUDA_TRIPLET_TYPES_HPP
#define CUDA_TRIPLET_TYPES_HPP

#include <cstdint>
#include <type_traits>
#include <vector>


struct GpuTripletKey
{
    std::uint8_t keys[3]{};
    std::uint8_t padding{};

    std::uint32_t distance_bins[3]{};
};


struct GpuProteinTripletRecord
{
    std::uint32_t indices[3]{};
};


// Sorted keys with CSR-style buckets: the half-open record range
// [offsets[i], offsets[i + 1]) belongs to keys[i]. Packing preserves CPU order.
struct GpuFlatTripletIndexHost
{
    std::vector<GpuTripletKey> keys;
    std::vector<std::uint32_t> offsets;
    std::vector<GpuProteinTripletRecord> records;
};


struct GpuFlatTripletIndexDevice
{
    GpuTripletKey* keys{nullptr};

    std::uint32_t* offsets{nullptr};

    GpuProteinTripletRecord* records{nullptr};

    std::uint32_t key_count{};
    std::uint32_t offset_count{};
    std::uint32_t record_count{};
};


static_assert(
    std::is_trivially_copyable_v<GpuTripletKey>
);

static_assert(
    std::is_trivially_copyable_v<
        GpuProteinTripletRecord
    >
);

static_assert(
    sizeof(GpuTripletKey) == 16
);

static_assert(
    sizeof(GpuProteinTripletRecord) == 12
);

// One lookup match before expansion into equivalent ligand vertex orders.
// Protein indices address the target-point array, not protein atom indices.
struct GpuTripletCandidate
{
    std::uint32_t ligand_triplet_index{};
    std::uint32_t protein_indices[3]{};
};

static_assert(
    std::is_trivially_copyable_v<
        GpuTripletCandidate
    >
);

static_assert(
    sizeof(GpuTripletCandidate) == 16
);

struct GpuLigandTripletOrders
{
    std::uint32_t equivalent_order_count{};
    std::uint32_t equivalent_orders[6][3]{};
};


struct GpuAlignmentCandidate
{
    std::uint32_t ligand_indices[3]{};
    std::uint32_t protein_indices[3]{};
};


static_assert(
    std::is_trivially_copyable_v<
        GpuLigandTripletOrders
    >
);

static_assert(
    std::is_trivially_copyable_v<
        GpuAlignmentCandidate
    >
);

static_assert(
    sizeof(GpuLigandTripletOrders) == 76
);

static_assert(
    sizeof(GpuAlignmentCandidate) == 24
);

struct GpuPoint3
{
    double x{};
    double y{};
    double z{};
};

static_assert(
    std::is_trivially_copyable_v<GpuPoint3>
);

static_assert(
    sizeof(GpuPoint3) == 24
);

struct GpuPointArrayDevice
{
    GpuPoint3* points{nullptr};
    std::uint32_t count{};
};

static_assert(
    std::is_trivially_copyable_v<
        GpuPointArrayDevice
    >
);

// Apply x' = R*x + t with a row-major rotation. For Kabsch outputs, valid
// reports SVD status or the quaternion norm guard, without a separate rank
// or finiteness check.
struct GpuRigidTransform
{
    double rotation[9]{};
    double translation[3]{};

    std::uint32_t valid{};
    std::uint32_t padding[3]{};
};

static_assert(
    std::is_trivially_copyable_v<
        GpuRigidTransform
    >
);

static_assert(
    sizeof(GpuRigidTransform) == 112
);

// Plain handle: returned allocations must be released with
// cuda_free_rigid_transforms(). Copying the handle creates a borrowed alias.
struct GpuRigidTransformArrayDevice
{
    GpuRigidTransform* transforms{nullptr};
    std::uint32_t count{};
};

#endif
