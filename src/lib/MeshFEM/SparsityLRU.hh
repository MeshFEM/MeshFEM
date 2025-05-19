////////////////////////////////////////////////////////////////////////////////
// SparsityLRU.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Implements a strategy for reducing the number of symbolic factorizations
//  needed for problems with occasionally changing sparsity patterns, e.g.,
//  for elasticity simulations with contact.
//
//  The idea is to retain up to a configurable number of previously occupied
//  entries in case they reappear in upcoming iterations. When this budget is
//  exceeded, the least recently used (LRU) entries are evicted the next time
//  the sparsity pattern needs to be rebuilt.
//
//  Author:  Julian Panetta (jpanetta), jpanetta@ucdavis.edu
//  Company:  University of California, Davis
//  Created:  05/11/2025 12:05:06
*///////////////////////////////////////////////////////////////////////////////
#ifndef SPARSITYLRU_HH
#define SPARSITYLRU_HH

#include <MeshFEM/newton_optimizer/NewtonHessian.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include <stdexcept>

struct SparsityLRU {
    // The full sparsity pattern consists of a (typically large) static part
    // and (typically small) dynamic part. Entries in the static part are
    // assigned a special age to indicate they should never be removed.
    static constexpr int STATIC_ENTRY_AGE = -1;
    static constexpr int EXPIRED = std::numeric_limits<int>::max();
    using SpMat = SuiteSparseMatrix; // Base class of BlockCSCHessianBase...
    using Index = typename SpMat::index_type;

    SparsityLRU(const SpMat &S_static)
        : m_S(S_static, /* sparsityOnly = */ true), m_entryAge(S_static.Ai.size(), STATIC_ENTRY_AGE) { }

    template<typename Predicate>
    void pruneEntries(const Predicate &shouldRemove) {
        if (m_S.Ax.size() != 0) throw std::runtime_error("SparsityLRU: pruneEntries called on non-sparsity-only matrix");
        size_t back = 0;
        Index colStart = m_S.Ap[0];
        for (Index j = 0; j < m_S.n; ++j) {
            Index colEnd = m_S.Ap[j + 1];
            for (Index ii = colStart; ii < colEnd; ++ii) {
                if (shouldRemove(ii)) continue;
                m_S.Ai[back] = m_S.Ai[ii];
                m_entryAge[back] = m_entryAge[ii];
                ++back;
            }
            m_S.Ap[j + 1] = back;
            colStart = colEnd;
        }

        m_S.nz = back;
        m_S.Ai.resize(back);
        m_entryAge.resize(back);
    }

    void resetAge(Index ii) {
        if (m_entryAge[ii] != STATIC_ENTRY_AGE)
            m_entryAge[ii] = 0;
    }

    void incrementAge(Index ii, int &maxAge) {
        if (m_entryAge[ii] != STATIC_ENTRY_AGE)
            maxAge = std::max(maxAge, ++m_entryAge[ii]);
    }

    // Returns the number of new entries added to the cache.
    // A return value of 0 means no new entries were added, and the cached
    // sparsity pattern can be reused.
    // A special value of `EXPIRED` indicates that the cache was
    // invalidated despite the absence of new entries because all of its
    // extra entries expired.
    Index update(const SpMat &S_dynamic) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("SparsityLRU.update");

        if (m_entryAge.size() != m_S.Ai.size()) throw std::runtime_error("SparsityLRU: age size mismatch");
        if ((S_dynamic.m != m_S.m) || (S_dynamic.n != m_S.n)) throw std::runtime_error("SparsityLRU: S_dynamic size mismatch");
        if (S_dynamic.symmetry_mode != m_S.symmetry_mode) throw std::runtime_error("Symmetry mode mismatch");

        // Fast path: if the dynamic sparsity pattern is empty, we can
        // just increment the ages of all non-static entries in the cache.
        int maxAge = STATIC_ENTRY_AGE;
        if (S_dynamic.nz == 0) {
            for (int &age : m_entryAge) {
                if (age != STATIC_ENTRY_AGE) {
                    ++age;
                    maxAge = std::max(maxAge, age);
                }
            }

            if (maxAge >= hardExpirationAge) {
                pruneEntries([this](Index i) { return m_entryAge[i] >= expirationAge; });
                return EXPIRED;
            }
            return 0;
        }

        // Merge entries from the dynamic sparsity pattern into the cache.
        // First, update ages and count how many new entries are needed in each column.
        Index totalCurrentEntries = 0; // Entries actually present in the current "static + dynamic" sparsity pattern
        Index totalNewEntries = 0;     // Current entries that are not already in the cache.
        m_oldEntryLocations.clear();   // Non-static entries that are not present in the new dynamic sparsity pattern
        std::vector<Index> newEntries(m_S.n, 0);
        // TODO: parallelize over the columns! (This requires reducing `maxAge`, `totalNewEntries`, and `oldEntryLocations` using TLS)
        for (Index j = 0; j < m_S.n; ++j) {
            Index ii_S =       m_S.Ap[j], ii_S_end =       m_S.Ap[j + 1];
            Index ii_D = S_dynamic.Ap[j], ii_D_end = S_dynamic.Ap[j + 1];

            // Handle entries that don't appear in `S_dynamic`
            // (increase their age and mark them as old)
            auto entryInSOnly = [&ii_S, &maxAge, j, this]() {
                incrementAge(ii_S, maxAge);
                if (m_entryAge[ii_S] != STATIC_ENTRY_AGE)
                    m_oldEntryLocations.emplace_back(j, ii_S);
                ++ii_S;
            };

            // Handle entries that don't appear in `S` yet
            // (count new entries)
            auto entryInDOnly = [&ii_D, &totalNewEntries, &totalCurrentEntries, &newEntries, j]() {
                ++newEntries[j]; ++totalNewEntries; ++totalCurrentEntries;
                ++ii_D;
            };

            while ((ii_S < ii_S_end) && (ii_D < ii_D_end)) {
                if (m_S.Ai[ii_S] == S_dynamic.Ai[ii_D]) {
                    resetAge(ii_S);
                    ++totalCurrentEntries;
                    ++ii_S; ++ii_D;
                }
                else if (m_S.Ai[ii_S] < S_dynamic.Ai[ii_D]) {
                    entryInSOnly();
                }
                else entryInDOnly();
            }

            // Handle remaining entries in S and D
            while (ii_S < ii_S_end) entryInSOnly();
            while (ii_D < ii_D_end) entryInDOnly();
        }

        // Semi-fast path: reuse (potentially pruned) cache if no new entries were added
        if (totalNewEntries == 0) {
            if (maxAge >= hardExpirationAge) {
                pruneEntries([this](Index i) { return m_entryAge[i] >= expirationAge; });
                return EXPIRED;
            }
            return 0;
        }

        // Doing a full rebuild because new entries were added!
        SpMat S_new(m_S.m, m_S.n);
        S_new.symmetry_mode = m_S.symmetry_mode;
        SpMat::InOrderBuilder builder(S_new, [&](Index *colSizes) {
            // Count the number of nonzeros in each column of the new sparsity cache.
            // Start with old + new entries...
            for (Index j = 0; j < m_S.n; ++j)
                colSizes[j] = m_S.col_nnz(j) + newEntries[j];

            // ... then determine how many of the old entries must be dropped.
            Index budget = totalCurrentEntries * entryCacheBudgetRatio;
            Index numOldEntries = m_oldEntryLocations.size();
            // Subtract the entries to be removed.
            if (numOldEntries > budget) {
                // Keep only the newest `budget` entries, implemented by sorting
                // entries non-descending by age.
                // TODO: use a better algorithm based on order statistics?
                // First, sort the entries non-descending by age.
                std::sort(m_oldEntryLocations.begin(), m_oldEntryLocations.end(),
                          [this](const EntryLoc &a, const EntryLoc &b) { return m_entryAge[a.ii] < m_entryAge[b.ii]; });
            }

            for (Index i = 0; i < numOldEntries; ++i) {
                const EntryLoc &loc = m_oldEntryLocations[i];
                if (i >= budget)
                    m_entryAge[loc.ii] = EXPIRED; // Mark the entries to be removed due to exceeding budget
                if (m_entryAge[loc.ii] >= expirationAge)
                    --colSizes[loc.j];
            }

        }, /* sparsityOnly = */ true);

        // Fill in the new cache and associated entry ages by doing
        // another merge of the cached and dynamic sparsity patterns.
        std::vector<int> new_ages;
        new_ages.reserve(S_new.Ai.size());

        for (Index j = 0; j < m_S.n; ++j) {
            Index ii_S =       m_S.Ap[j], ii_S_end =       m_S.Ap[j + 1];
            Index ii_D = S_dynamic.Ap[j], ii_D_end = S_dynamic.Ap[j + 1];

            auto entryInSOnly = [&builder, &ii_S, &new_ages, j, this]() {
                if (m_entryAge[ii_S] < expirationAge) {
                    builder.insert(m_S.Ai[ii_S], j);
                    new_ages.push_back(m_entryAge[ii_S]);
                }
                ++ii_S;
            };

            auto entryInDOnly = [&builder, &ii_D, &new_ages, j, &S_dynamic]() {
                builder.insert(S_dynamic.Ai[ii_D], j);
                new_ages.push_back(0);
                ++ii_D;
            };

            while ((ii_S < ii_S_end) && (ii_D < ii_D_end)) {
                if (m_S.Ai[ii_S] == S_dynamic.Ai[ii_D]) {
                    builder.insert(m_S.Ai[ii_S], j);
                    new_ages.push_back(std::min(0, m_entryAge[ii_S])); // Make sure negative ages (e.g., STATIC_ENTRY_AGE) are retained!
                    ++ii_S; ++ii_D;
                }
                else if (m_S.Ai[ii_S] < S_dynamic.Ai[ii_D])
                    entryInSOnly();
                else entryInDOnly();
            }

            // Handle remaining entries in S and D
            while (ii_S < ii_S_end) entryInSOnly();
            while (ii_D < ii_D_end) entryInDOnly();
        }

        if (new_ages.size() != S_new.Ai.size())
            throw std::runtime_error("updated new ages size mismatch: " + std::to_string(new_ages.size()) + " != " + std::to_string(S_new.Ai.size()));

        // Verify that S_new has all of its diagonal entries.
        for (Index j = 0; j < S_new.n; ++j) {
            Index ii = S_new.Ap[j + 1] - 1;
            if (S_new.Ai[ii] != j) throw std::runtime_error("SparsityLRU: missing diagonal entry in new sparsity pattern");
        }

        m_S = std::move(S_new);
        m_entryAge = std::move(new_ages);

        return totalNewEntries;
    }

    const std::vector<int> &entryAges() const { return m_entryAge; }

    const SpMat &operator*() const { return m_S; }
    const SpMat *operator->() const { return &m_S; }

    // Maximum number of old entries to retain the the cache each time it is
    // rebuilt due to new entries appearing.
    double entryCacheBudgetRatio = 0.01;

    // Entries older than this are removed regardless of whether the budget is exceeded.
    int expirationAge = 25;
    int hardExpirationAge = 50; // Entries older than this trigger a rebuild.

private:
    SpMat m_S;
    std::vector<int> m_entryAge;
    struct EntryLoc {
        EntryLoc(Index j_, Index ii_) : j(j_), ii(ii_) { }
        Index j, ii; // Column index and location in `Ai`, respectively
    };
    std::vector<EntryLoc> m_oldEntryLocations;
};

#endif /* end of include guard: SPARSITYLRU_HH */
