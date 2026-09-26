module SharedLowRankGRS

using Random
using LinearAlgebra
using Statistics
using Printf

export FINAL_CONFIG, QUICK_CONFIG,
       check_camra_files,
       load_rating_file, load_group_members, normalize_group_members,
       split_gi_train_validation, build_gu_data, initialize_factors,
       train_shared_lowrank!, pretrain_shared_lowrank,
       build_improved_consensus_supports,
       improved_consensus_penalty_and_gradient,
       apply_improved_consensus_step!,
       load_group_negative_file, build_group_positive_sets,
       build_validation_ranking_data, evaluate_group_ranking_group_scale,
       finetune_group_scale_bias_listwise!,
       run_single_seed, run_five_seeds, summarize_results

# ------------------------------------------------------------------
# FINAL PAPER SETTINGS
# ------------------------------------------------------------------
FINAL_CONFIG = (
    rank = 32,
    pretrain_epochs = 300,
    ranking_epochs = 300,
    steps_per_epoch = 20_000,

    pretrain_lr = 0.005,
    ranking_lr = 1e-3,

    alpha = 1.0,       # GU contribution
    beta = 1.0,        # GI contribution
    lambda = 1e-5,     # latent-factor L2 regularization

    rho = 1.0,
    tau = 0.2,

    candidate_pool = 100,
    num_negatives = 20,

    gu_negatives_per_positive = 4,

    val_fraction = 0.10,
    split_seed = 2026,

    stage1_patience = 15,
    stage1_min_delta = 1e-4,

    lambda_alpha = 0.05,
    stage2_patience = 20,
)

# Short test configuration
QUICK_CONFIG = merge(
    FINAL_CONFIG,
    (
        pretrain_epochs = 10,
        ranking_epochs = 5,
        steps_per_epoch = 2_000,
    ),
)

#-------------------------------------------------------------------------
# 2. Data loading and relation construction
#-------------------------------------------------------------------------

const CAMRA_REQUIRED_FILES = [
    "userRatingTrain.txt",
    "groupRatingTrain.txt",
    "groupRatingTest.txt",
    "groupMember.txt",
    "groupRatingNegative.txt",
]

function check_camra_files(data_dir)
    missing = [f for f in CAMRA_REQUIRED_FILES if !isfile(joinpath(data_dir, f))]
    isempty(missing) || error("Missing CAMRa2011 files: " * join(missing, ", "))
    return true
end

# Load a CAMRa rating file and convert IDs to Julia's 1-based indexing.
function load_rating_file(path::AbstractString)
    first_ids = Int[]
    second_ids = Int[]
    ratings = Float64[]

    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        parts = split(s)

        push!(first_ids, parse(Int, parts[1]) + 1)
        push!(second_ids, parse(Int, parts[2]) + 1)
        push!(ratings, parse(Float64, parts[3]) / 100.0)
    end

    return first_ids, second_ids, ratings
end


# Load group membership information from the CAMRa group-member file.
function load_group_members(path::AbstractString)
    group_members = Dict{Int, Vector{Int}}()

    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        parts = split(s)

        group_id = parse(Int, parts[1])
        members = parse.(Int, split(parts[2], ","))

        group_members[group_id] = members
    end

    return group_members
end


# Convert group and user IDs to 1-based indexing and return members by group.
function normalize_group_members(group_members_raw)
    group_keys = Int.(collect(keys(group_members_raw)))

    group_offset = minimum(group_keys) == 0 ? 1 : 0
    max_group = maximum(group_keys) + group_offset

    all_members = Int[]
    for members in values(group_members_raw)
        append!(all_members, Int.(members))
    end

    member_offset =
        !isempty(all_members) && minimum(all_members) == 0 ? 1 : 0

    group_members = [Int[] for _ in 1:max_group]

    for (g_raw, members_raw) in group_members_raw
        g = Int(g_raw) + group_offset
        group_members[g] = Int.(members_raw) .+ member_offset
    end

    return group_members
end


# Split group-item interactions into reproducible training and validation sets by group.
function split_gi_train_validation(
    groups,
    items,
    ratings;
    val_fraction=0.10,
    seed=2026,
)
    rng = MersenneTwister(seed)

    by_group = Dict{Int, Vector{Int}}()

    for idx in eachindex(groups)
        push!(get!(by_group, groups[idx], Int[]), idx)
    end

    train_idx = Int[]
    val_idx = Int[]

    for g in sort(collect(keys(by_group)))
        idxs = copy(by_group[g])
        shuffle!(rng, idxs)

        n = length(idxs)

        if n <= 1
            append!(train_idx, idxs)
            continue
        end

        nval = clamp(
            round(Int, val_fraction * n),
            1,
            n - 1,
        )

        append!(val_idx, idxs[1:nval])
        append!(train_idx, idxs[nval+1:end])
    end

    sort!(train_idx)
    sort!(val_idx)

    train = (
        groups = groups[train_idx],
        items = items[train_idx],
        ratings = ratings[train_idx],
    )

    validation = (
        groups = groups[val_idx],
        items = items[val_idx],
        ratings = ratings[val_idx],
    )

    return train, validation
end


# Build positive and sampled-negative group-user membership examples.
function build_gu_data(
    group_members,
    num_users;
    negatives_per_positive=4,
    seed=2026,
)
    rng = MersenneTwister(seed)

    groups = Int[]
    users = Int[]
    labels = Float64[]

    all_users = collect(1:num_users)

    for g in eachindex(group_members)
        members = group_members[g]
        isempty(members) && continue

        member_set = Set(members)

        # Positive memberships.
        for u in members
            push!(groups, g)
            push!(users, u)
            push!(labels, 1.0)
        end

        # Sampled non-members.
        nonmembers = [
            u for u in all_users
            if !(u in member_set)
        ]

        nneg = min(
            negatives_per_positive * length(members),
            length(nonmembers),
        )

        if nneg > 0
            chosen = randperm(rng, length(nonmembers))[1:nneg]

            for idx in chosen
                push!(groups, g)
                push!(users, nonmembers[idx])
                push!(labels, 0.0)
            end
        end
    end

    return (
        groups = groups,
        users = users,
        labels = labels,
    )
end

#--------------------------------------------------------
# 3. Stage 1 — shared low-rank representation learning
#-------------------------------------------------------
sigmoid(x::Real) =
    1.0 / (1.0 + exp(-clamp(x, -30.0, 30.0)))


# Initialize user, group, and item latent factors with a fixed random seed.
function initialize_factors(
    num_users,
    num_groups,
    num_items,
    rank;
    scale=0.1,
    seed=42,
)
    rng = MersenneTwister(seed)

    U = scale .* randn(rng, num_users, rank)
    H = scale .* randn(rng, num_groups, rank)
    V = scale .* randn(rng, num_items, rank)

    return U, H, V
end


# Compute mean squared error for observed user-item ratings.
function ui_mse(U, V, users, items, ratings)
    s = 0.0

    for t in eachindex(ratings)
        pred =
            dot(
                @view(U[users[t], :]),
                @view(V[items[t], :]),
            )

        s += (ratings[t] - pred)^2
    end

    return s / length(ratings)
end


# Compute mean squared error for observed group-item ratings.
function gi_mse(H, V, groups, items, ratings)
    s = 0.0

    for t in eachindex(ratings)
        pred =
            dot(
                @view(H[groups[t], :]),
                @view(V[items[t], :]),
            )

        s += (ratings[t] - pred)^2
    end

    return s / length(ratings)
end


# Compute binary cross-entropy for group-user membership prediction.
function gu_bce(H, U, groups, users, labels)
    s = 0.0
    eps_prob = 1e-8

    for t in eachindex(labels)
        p = clamp(
            sigmoid(
                dot(
                    @view(H[groups[t], :]),
                    @view(U[users[t], :]),
                )
            ),
            eps_prob,
            1.0 - eps_prob,
        )

        y = labels[t]

        s -=
            y * log(p) +
            (1.0 - y) * log(1.0 - p)
    end

    return s / length(labels)
end


# Compute RMSE and MAE for group-item rating prediction.
function group_rating_metrics(
    H,
    V,
    groups,
    items,
    ratings,
)
    se = 0.0
    ae = 0.0

    for t in eachindex(ratings)
        pred =
            dot(
                @view(H[groups[t], :]),
                @view(V[items[t], :]),
            )

        e = ratings[t] - pred

        se += e^2
        ae += abs(e)
    end

    return (
        rmse = sqrt(se / length(ratings)),
        mae = ae / length(ratings),
    )
end


# Pre-generate sampled training indices so experiments are reproducible.
function make_fixed_samples(
    n_ui,
    n_gu,
    n_gi;
    epochs=50,
    steps_per_epoch=20_000,
    seed=123,
)
    rng = MersenneTwister(seed)

    ui_samples = [
        rand(rng, 1:n_ui, steps_per_epoch)
        for _ in 1:epochs
    ]

    gu_samples = [
        rand(rng, 1:n_gu, steps_per_epoch)
        for _ in 1:epochs
    ]

    gi_samples = [
        rand(rng, 1:n_gi, steps_per_epoch)
        for _ in 1:epochs
    ]

    return ui_samples, gu_samples, gi_samples
end


# Train the shared user-group-item low-rank representation and keep the best validation checkpoint.
function train_shared_lowrank!(
    U,
    H,
    V,

    ui_users,
    ui_items,
    ui_ratings,

    gu_groups,
    gu_users,
    gu_labels,

    gi_groups,
    gi_items,
    gi_ratings,

    group_members,

    ui_samples,
    gu_samples,
    gi_samples;

    lr=0.005,
    alpha=1.0,
    beta=1.0,
    lambda=1e-5,

    gamma=0.0,
    eta=0.90,
    mu=1.0,

    validation_data=nothing,
    patience=15,
    min_delta=1e-4,
    verbose=true,
)
    history = NamedTuple[]

    best_val_rmse = Inf
    best_epoch = 0

    best_U = copy(U)
    best_H = copy(H)
    best_V = copy(V)

    epochs_without_improvement = 0

    for epoch in eachindex(ui_samples)
        # Relation-wise SGD.
        for step in eachindex(ui_samples[epoch])
            # UI update: R_UI ~ U V'.
            t = ui_samples[epoch][step]
            u = ui_users[t]
            i = ui_items[t]
            y = ui_ratings[t]

            uvec = copy(@view U[u, :])
            ivec = copy(@view V[i, :])
            err = y - dot(uvec, ivec)

            U[u, :] .+= lr .* (2.0 .* err .* ivec .- lambda .* uvec)
            V[i, :] .+= lr .* (2.0 .* err .* uvec .- lambda .* ivec)

            # GU update: A_GU ~ sigmoid(H U').
            t = gu_samples[epoch][step]
            g = gu_groups[t]
            u = gu_users[t]
            y = gu_labels[t]

            hvec = copy(@view H[g, :])
            uvec = copy(@view U[u, :])
            coeff = y - sigmoid(dot(hvec, uvec))

            H[g, :] .+= lr .* alpha .* (coeff .* uvec .- lambda .* hvec)
            U[u, :] .+= lr .* alpha .* (coeff .* hvec .- lambda .* uvec)

            # GI update: R_GI ~ H V'.
            t = gi_samples[epoch][step]
            g = gi_groups[t]
            i = gi_items[t]
            y = gi_ratings[t]

            hvec = copy(@view H[g, :])
            ivec = copy(@view V[i, :])
            err = y - dot(hvec, ivec)

            H[g, :] .+= lr .* beta .* (2.0 .* err .* ivec .- lambda .* hvec)
            V[i, :] .+= lr .* beta .* (2.0 .* err .* hvec .- lambda .* ivec)
        end

        # Consensus-aware regularisation is applied once per epoch.
        consensus_penalty = 0.0
        if gamma > 0.0
            consensus_penalty = apply_improved_consensus_step!(
                H,
                U,
                group_members;
                lr=lr,
                gamma=gamma,
                eta=eta,
                mu=mu,
            )
        end

        # Metrics are measured after all Stage-1 updates for this epoch.
        train_ui = ui_mse(U, V, ui_users, ui_items, ui_ratings)
        train_gu = gu_bce(H, U, gu_groups, gu_users, gu_labels)
        train_gi = gi_mse(H, V, gi_groups, gi_items, gi_ratings)

        val_rmse = NaN
        val_mae = NaN

        if validation_data !== nothing
            vm = group_rating_metrics(
                H,
                V,
                validation_data.groups,
                validation_data.items,
                validation_data.ratings,
            )

            val_rmse = vm.rmse
            val_mae = vm.mae

            if val_rmse < best_val_rmse - min_delta
                best_val_rmse = val_rmse
                best_epoch = epoch
                best_U .= U
                best_H .= H
                best_V .= V
                epochs_without_improvement = 0
            else
                epochs_without_improvement += 1
            end
        end

        push!(
            history,
            (
                epoch=epoch,
                ui_mse=train_ui,
                gu_bce=train_gu,
                gi_mse=train_gi,
                consensus_penalty=consensus_penalty,
                val_rmse=val_rmse,
                val_mae=val_mae,
            ),
        )

        if verbose
            @printf(
                "Stage 1 epoch %d | UI=%.4f | GU=%.4f | GI=%.4f | Cons=%.4f | Val RMSE=%.4f | Val MAE=%.4f
",
                epoch,
                train_ui,
                train_gu,
                train_gi,
                consensus_penalty,
                val_rmse,
                val_mae,
            )
        end

        if validation_data !== nothing && epochs_without_improvement >= patience
            verbose && println(
                "Stage 1 early stopping at epoch $epoch | " *
                "best epoch=$best_epoch | " *
                "best val RMSE=$(round(best_val_rmse, digits=6))"
            )
            break
        end
    end

    return (
        history=history,
        best_U=best_U,
        best_H=best_H,
        best_V=best_V,
        best_epoch=best_epoch,
        best_val_rmse=best_val_rmse,
    )
end

#---------------------------------------------------------------
# 4. Stage-1 wrapper
#---------------------------------------------------------------

# Run Stage 1: load data, build relations, train shared factors, and evaluate rating prediction.
function pretrain_shared_lowrank(
    data_dir;
    model_seed=42,
    config=FINAL_CONFIG,
    verbose=false,
    gamma=0.0,
    eta=0.90,
    mu=1.0,
)
    check_camra_files(data_dir)

    ui_users, ui_items, ui_ratings =
        load_rating_file(joinpath(data_dir, "userRatingTrain.txt"))

    gi_groups_all, gi_items_all, gi_ratings_all =
        load_rating_file(joinpath(data_dir, "groupRatingTrain.txt"))

    test_groups, test_items, test_ratings =
        load_rating_file(joinpath(data_dir, "groupRatingTest.txt"))

    group_members = normalize_group_members(
        load_group_members(joinpath(data_dir, "groupMember.txt"))
    )

    all_member_ids = isempty(group_members) ? Int[] : reduce(vcat, group_members)
    num_users = maximum(vcat(ui_users, all_member_ids))
    num_groups = length(group_members)
    num_items = maximum(vcat(ui_items, gi_items_all, test_items))

    gu_data = build_gu_data(
        group_members,
        num_users;
        negatives_per_positive=config.gu_negatives_per_positive,
        seed=config.split_seed,
    )

    gi_train, gi_val = split_gi_train_validation(
        gi_groups_all,
        gi_items_all,
        gi_ratings_all;
        val_fraction=config.val_fraction,
        seed=config.split_seed,
    )

    U, H, V = initialize_factors(
        num_users,
        num_groups,
        num_items,
        config.rank;
        seed=model_seed,
    )

    ui_samples, gu_samples, gi_samples = make_fixed_samples(
        length(ui_ratings),
        length(gu_data.labels),
        length(gi_train.ratings);
        epochs=config.pretrain_epochs,
        steps_per_epoch=config.steps_per_epoch,
        seed=model_seed + 1000,
    )

    pretrain = train_shared_lowrank!(
        U,
        H,
        V,
        ui_users,
        ui_items,
        ui_ratings,
        gu_data.groups,
        gu_data.users,
        gu_data.labels,
        gi_train.groups,
        gi_train.items,
        gi_train.ratings,
        group_members,
        ui_samples,
        gu_samples,
        gi_samples;
        lr=config.pretrain_lr,
        alpha=config.alpha,
        beta=config.beta,
        lambda=config.lambda,
        gamma=gamma,
        eta=eta,
        mu=mu,
        validation_data=gi_val,
        patience=config.stage1_patience,
        min_delta=config.stage1_min_delta,
        verbose=verbose,
    )

    U0 = copy(pretrain.best_U)
    H0 = copy(pretrain.best_H)
    V0 = copy(pretrain.best_V)

    rating = group_rating_metrics(H0, V0, test_groups, test_items, test_ratings)

    @printf(
        "Stage 1 | seed=%d | best epoch=%d | val RMSE=%.6f | test RMSE=%.6f | test MAE=%.6f
",
        model_seed,
        pretrain.best_epoch,
        pretrain.best_val_rmse,
        rating.rmse,
        rating.mae,
    )

    return (
        pretrain_U=U0,
        pretrain_H=H0,
        pretrain_V=V0,
        pretrain_epoch=pretrain.best_epoch,
        best_val_rmse=pretrain.best_val_rmse,
        test_rmse=rating.rmse,
        test_mae=rating.mae,
        history=pretrain.history,
        gi_train=gi_train,
        gi_val=gi_val,
        gi_groups_all=gi_groups_all,
        gi_items_all=gi_items_all,
        gi_ratings_all=gi_ratings_all,
        test_groups=test_groups,
        test_items=test_items,
        test_ratings=test_ratings,
        group_members=group_members,
        num_users=num_users,
        num_groups=num_groups,
        num_items=num_items,
        gamma=gamma,
        eta=eta,
        mu=mu,
    )
end

# -------------------------------------------------------------------------
# 5. Ranking protocol and leakage-safe validation candidates
# ---------------------------------------------------------------------------
# Load the official CAMRa group-ranking positives and negative candidate items.

function load_group_negative_file(path::AbstractString)
    groups = Int[]
    positives = Int[]
    negatives = Vector{Vector{Int}}()

    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue

        parts = split(s)

        pair_clean =
            replace(
                parts[1],
                "(" => "",
                ")" => "",
            )

        pair_parts =
            split(
                pair_clean,
                ",",
            )

        push!(
            groups,
            parse(Int, pair_parts[1]) + 1,
        )

        push!(
            positives,
            parse(Int, pair_parts[2]) + 1,
        )

        push!(
            negatives,
            parse.(Int, parts[2:end]) .+ 1,
        )
    end

    return groups, positives, negatives
end


# Store known positive items for each group for negative-sampling exclusions.
function build_group_positive_sets(
    groups,
    items,
    num_groups,
)
    positives = [
        Set{Int}()
        for _ in 1:num_groups
    ]

    for t in eachindex(groups)
        push!(
            positives[groups[t]],
            items[t],
        )
    end

    return positives
end


# Create leakage-safe validation ranking candidates with sampled negatives.
function build_validation_ranking_data(
    val_groups,
    val_items,
    num_items,
    group_positive_sets;
    negatives_per_positive=100,
    seed=3030,
)
    rng = MersenneTwister(seed)

    negative_items =
        Vector{Vector{Int}}(
            undef,
            length(val_groups),
        )

    for q in eachindex(val_groups)
        g = val_groups[q]
        positive = val_items[q]

        negs = Int[]
        used = Set{Int}()

        while length(negs) < negatives_per_positive
            j = rand(rng, 1:num_items)

            if j != positive &&
               !(j in group_positive_sets[g]) &&
               !(j in used)

                push!(negs, j)
                push!(used, j)
            end
        end

        negative_items[q] = negs
    end

    return (
        groups = val_groups,
        positives = val_items,
        negatives = negative_items,
    )
end


# Evaluate HR@K and NDCG@K using the calibrated group-ranking score.
function evaluate_group_ranking_group_scale(
    H,
    V,
    group_scale,
    item_bias,
    groups,
    positive_items,
    negative_items;
    tau=0.2,
    Ks=[5, 10, 20],
)
    num_queries = length(groups)

    hits =
        Dict(
            K => 0.0
            for K in Ks
        )

    ndcgs =
        Dict(
            K => 0.0
            for K in Ks
        )

    for q in eachindex(groups)
        g = groups[q]
        positive = positive_items[q]

        candidates =
            vcat(
                positive,
                negative_items[q],
            )

        h =
            @view H[g, :]

        scores = [
            group_scale[g] *
            (
                dot(
                    h,
                    @view(V[i, :]),
                ) / tau
            ) +
            item_bias[i]
            for i in candidates
        ]

        order =
            sortperm(
                scores;
                rev=true,
            )

        # Candidate position 1 is the positive item.
        positive_rank =
            findfirst(
                ==(1),
                order,
            )

        for K in Ks
            if positive_rank <= K
                hits[K] += 1.0

                ndcgs[K] +=
                    1.0 /
                    log2(
                        positive_rank + 1
                    )
            end
        end
    end

    return (
        HR =
            Dict(
                K => hits[K] / num_queries
                for K in Ks
            ),

        NDCG =
            Dict(
                K => ndcgs[K] / num_queries
                for K in Ks
            ),
    )
end

# -------------------------------------------------------------------------
# 6. Stage 2 — group-scale + item-bias listwise ranking calibration
# -------------------------------------------------------------------------

# Run Stage 2: learn group scales and item biases with hard-negative listwise training.
function finetune_group_scale_bias_listwise!(
    H,
    V,

    gi_groups,
    gi_items,

    gi_samples,
    train_positive_sets,
    validation_ranking_data;

    lr=1e-3,
    rho=1.0,
    tau=0.2,

    candidate_pool=100,
    num_negatives=20,

    ranking_K=10,
    lambda_alpha=0.05,

    patience=20,

    seed=9999,
    verbose=true,
)
    rng = MersenneTwister(seed)

    num_groups = size(H, 1)
    num_items = size(V, 1)

    # H and V remain frozen.
    group_scale =
        ones(
            Float64,
            num_groups,
        )

    item_bias =
        zeros(
            Float64,
            num_items,
        )

    initial_rank =
        evaluate_group_ranking_group_scale(
            H,
            V,
            group_scale,
            item_bias,

            validation_ranking_data.groups,
            validation_ranking_data.positives,
            validation_ranking_data.negatives;

            tau=tau,
            Ks=[ranking_K],
        )

    best_epoch = 0
    best_ndcg =
        initial_rank.NDCG[ranking_K]

    best_group_scale =
        copy(group_scale)

    best_item_bias =
        copy(item_bias)

    history = NamedTuple[
        (
            epoch = 0,
            val_HR =
                initial_rank.HR[ranking_K],
            val_NDCG =
                initial_rank.NDCG[ranking_K],
        )
    ]

    epochs_without_improvement = 0

    for epoch in eachindex(gi_samples)

        for step in eachindex(gi_samples[epoch])
            t = gi_samples[epoch][step]

            g = gi_groups[t]
            ipos = gi_items[t]

            # ----------------------------------------------------
            # Random candidate pool excluding GI training positives
            # ----------------------------------------------------
            candidates = Int[]

            while length(candidates) < candidate_pool
                j = rand(rng, 1:num_items)

                if !(j in train_positive_sets[g]) &&
                   !(j in candidates)

                    push!(candidates, j)
                end
            end

            h =
                @view H[g, :]

            candidate_scores = [
                group_scale[g] *
                (
                    dot(
                        h,
                        @view(V[j, :]),
                    ) / tau
                ) +
                item_bias[j]
                for j in candidates
            ]

            order =
                sortperm(
                    candidate_scores;
                    rev=true,
                )

            kkeep =
                min(
                    num_negatives,
                    length(order),
                )

            negatives =
                candidates[
                    order[1:kkeep]
                ]

            # ----------------------------------------------------
            # Positive first, then selected hard negatives
            # ----------------------------------------------------
            items =
                vcat(
                    ipos,
                    negatives,
                )

            base_scores = [
                dot(
                    h,
                    @view(V[item, :]),
                ) / tau
                for item in items
            ]

            scores = [
                group_scale[g] *
                base_scores[k] +
                item_bias[items[k]]
                for k in eachindex(items)
            ]

            # Stable listwise softmax cross-entropy.
            score_max =
                maximum(scores)

            exp_scores =
                exp.(
                    scores .-
                    score_max
                )

            probs =
                exp_scores ./
                sum(exp_scores)

            score_grads =
                copy(probs)

            # Candidate 1 is the positive item.
            score_grads[1] -= 1.0

            # ----------------------------------------------------
            # Item-bias updates
            # ----------------------------------------------------
            item_bias[ipos] -=
                lr *
                rho *
                score_grads[1]

            for k in eachindex(negatives)
                item_bias[negatives[k]] -=
                    lr *
                    rho *
                    score_grads[k + 1]
            end

            # ----------------------------------------------------
            # Group-specific scale update
            # ----------------------------------------------------
            grad_scale = 0.0

            for k in eachindex(items)
                grad_scale +=
                    score_grads[k] *
                    base_scores[k]
            end

            grad_scale +=
                2.0 *
                lambda_alpha *
                (
                    group_scale[g] -
                    1.0
                )

            group_scale[g] -=
                lr *
                rho *
                grad_scale
        end

        ranking =
            evaluate_group_ranking_group_scale(
                H,
                V,
                group_scale,
                item_bias,

                validation_ranking_data.groups,
                validation_ranking_data.positives,
                validation_ranking_data.negatives;

                tau=tau,
                Ks=[ranking_K],
            )

        val_hr =
            ranking.HR[ranking_K]

        val_ndcg =
            ranking.NDCG[ranking_K]

        push!(
            history,
            (
                epoch = epoch,
                val_HR = val_hr,
                val_NDCG = val_ndcg,
            ),
        )

        if val_ndcg > best_ndcg
            best_ndcg = val_ndcg
            best_epoch = epoch

            best_group_scale .=
                group_scale

            best_item_bias .=
                item_bias

            epochs_without_improvement = 0
        else
            epochs_without_improvement += 1
        end

        if verbose
            @printf(
                "Stage 2 epoch %d | HR@%d=%.6f | NDCG@%d=%.6f\n",
                epoch,
                ranking_K,
                val_hr,
                ranking_K,
                val_ndcg,
            )
        end

        if epochs_without_improvement >= patience
            verbose && println(
                "Stage 2 early stopping at epoch $epoch | " *
                "best epoch=$best_epoch | " *
                "best NDCG@$ranking_K=$(round(best_ndcg, digits=6))"
            )

            break
        end
    end

    return (
        history = history,
        best_epoch = best_epoch,
        best_ndcg = best_ndcg,
        best_group_scale = best_group_scale,
        best_item_bias = best_item_bias,
    )
end

# -----------------------------------------------------------------------
# 7. Consensus-aware extension
# -----------------------------------------------------------------------

# Build the dominant member-preference subspace and direction-wise consensus.
function build_improved_consensus_supports(
    U,
    group_members;
    eta=0.90,
    eps_consensus=1e-8,
)
    supports = Vector{NamedTuple}(undef, length(group_members))

    for g in eachindex(group_members)
        members = group_members[g]

        if isempty(members)
            supports[g] = (
                Q = zeros(Float64, size(U, 2), 0),
                consensus = Float64[],
            )
            continue
        end

        Ug = Matrix(U[members, :])
        F = svd(Ug; full=false)

        sigma2 = F.S .^ 2
        total_energy = sum(sigma2)

        if total_energy <= eps_consensus
            supports[g] = (
                Q = zeros(Float64, size(U, 2), 0),
                consensus = Float64[],
            )
            continue
        end

        cumulative = cumsum(sigma2) ./ total_energy
        q = findfirst(x -> x >= eta, cumulative)
        q === nothing && (q = length(F.S))

        Qg = Matrix(F.V[:, 1:q])
        consensus = zeros(Float64, q)

        for k in 1:q
            direction = @view Qg[:, k]
            projections = Ug * direction

            consensus[k] =
                abs(sum(projections)) /
                (
                    sum(abs.(projections)) +
                    eps_consensus
                )
        end

        supports[g] = (
            Q = Qg,
            consensus = consensus,
        )
    end

    return supports
end


# Compute the improved consensus penalty and gradient for one group vector.
function improved_consensus_penalty_and_gradient(
    h,
    Qg,
    consensus;
    mu=1.0,
)
    if size(Qg, 2) == 0
        return (
            penalty = dot(h, h),
            gradient = 2.0 .* h,
        )
    end

    z = Qg' * h
    projected = Qg * z
    outside = h - projected

    outside_penalty = dot(outside, outside)
    disagreement = 1.0 .- consensus

    inside_penalty =
        mu *
        sum(
            disagreement .* (z .^ 2)
        )

    gradient =
        2.0 .* outside +
        2.0 *
        mu *
        (
            Qg *
            (
                disagreement .* z
            )
        )

    return (
        penalty =
            outside_penalty +
            inside_penalty,
        gradient = gradient,
    )
end

# -----------------------------------------------------------------------
# Consensus experiment result
# -----------------------------------------------------------------------



# Apply one improved consensus-aware update to every group representation.
function apply_improved_consensus_step!(
    H,
    U,
    group_members;
    lr,
    gamma=0.01,
    eta=0.90,
    mu=1.0,
    eps_consensus=1e-8,
)
    gamma <= 0.0 && return 0.0

    supports = build_improved_consensus_supports(
        U,
        group_members;
        eta=eta,
        eps_consensus=eps_consensus,
    )

    total_penalty = 0.0

    for g in axes(H, 1)
        support = supports[g]
        h = copy(@view H[g, :])

        result = improved_consensus_penalty_and_gradient(
            h,
            support.Q,
            support.consensus;
            mu=mu,
        )

        H[g, :] .-= lr .* gamma .* result.gradient
        total_penalty += result.penalty
    end

    return total_penalty
end

# ------------------------------------------------------------------------
# 8. Complete single-seed experiment
# ------------------------------------------------------------------------

# Run the complete two-stage experiment for one random seed.
function run_single_seed(
    data_dir;
    model_seed=42,
    config=FINAL_CONFIG,
    verbose=false,
    gamma=0.01,
    eta=0.90,
    mu=1.0,
)
    pretrained = pretrain_shared_lowrank(
        data_dir;
        model_seed=model_seed,
        config=config,
        verbose=verbose,
        gamma=gamma,
        eta=eta,
        mu=mu,
    )

    H0 = pretrained.pretrain_H
    V0 = pretrained.pretrain_V

    train_positive_sets = build_group_positive_sets(
        pretrained.gi_train.groups,
        pretrained.gi_train.items,
        pretrained.num_groups,
    )

    all_positive_sets = build_group_positive_sets(
        pretrained.gi_groups_all,
        pretrained.gi_items_all,
        pretrained.num_groups,
    )

    validation_ranking_data = build_validation_ranking_data(
        pretrained.gi_val.groups,
        pretrained.gi_val.items,
        pretrained.num_items,
        all_positive_sets;
        negatives_per_positive=100,
        seed=3030,
    )

    _, _, gi_samples = make_fixed_samples(
        1,
        1,
        length(pretrained.gi_train.ratings);
        epochs=config.ranking_epochs,
        steps_per_epoch=config.steps_per_epoch,
        seed=model_seed + 9000,
    )

    finetune = finetune_group_scale_bias_listwise!(
        H0,
        V0,
        pretrained.gi_train.groups,
        pretrained.gi_train.items,
        gi_samples,
        train_positive_sets,
        validation_ranking_data;
        lr=config.ranking_lr,
        rho=config.rho,
        tau=config.tau,
        candidate_pool=config.candidate_pool,
        num_negatives=config.num_negatives,
        ranking_K=10,
        lambda_alpha=config.lambda_alpha,
        patience=config.stage2_patience,
        seed=model_seed + 5000,
        verbose=verbose,
    )

    rank_groups, rank_positive_items, rank_negative_items = load_group_negative_file(
        joinpath(data_dir, "groupRatingNegative.txt")
    )

    final_ranking = evaluate_group_ranking_group_scale(
        H0,
        V0,
        finetune.best_group_scale,
        finetune.best_item_bias,
        rank_groups,
        rank_positive_items,
        rank_negative_items;
        tau=config.tau,
        Ks=[5, 10, 20],
    )

    println("FINAL MODEL - SEED $model_seed")
    println("Consensus gamma = ", gamma)
    println("Consensus eta   = ", eta)
    println("Consensus mu    = ", mu)

    @printf("Stage-1 best epoch       = %d\n", pretrained.pretrain_epoch)
    @printf("Stage-1 best Val RMSE    = %.6f\n", pretrained.best_val_rmse)
    @printf("RMSE                     = %.6f\n", pretrained.test_rmse)
    @printf("MAE                      = %.6f\n", pretrained.test_mae)
    @printf("HR@5                     = %.6f\n", final_ranking.HR[5])
    @printf("HR@10                    = %.6f\n", final_ranking.HR[10])
    @printf("HR@20                    = %.6f\n", final_ranking.HR[20])
    @printf("NDCG@5                   = %.6f\n", final_ranking.NDCG[5])
    @printf("NDCG@10                  = %.6f\n", final_ranking.NDCG[10])
    @printf("NDCG@20                  = %.6f\n", final_ranking.NDCG[20])
    @printf(
        "Stage-2 best Val NDCG@10 = %.6f at epoch %d\n",
        finetune.best_ndcg,
        finetune.best_epoch,
    )

    return (
        seed=model_seed,
        gamma=gamma,
        eta=eta,
        mu=mu,
        RMSE=pretrained.test_rmse,
        MAE=pretrained.test_mae,
        HR5=final_ranking.HR[5],
        HR10=final_ranking.HR[10],
        HR20=final_ranking.HR[20],
        NDCG5=final_ranking.NDCG[5],
        NDCG10=final_ranking.NDCG[10],
        NDCG20=final_ranking.NDCG[20],
        best_val_rmse=pretrained.best_val_rmse,
        pretrain_epoch=pretrained.pretrain_epoch,
        best_val_ndcg10=finetune.best_ndcg,
        best_ranking_epoch=finetune.best_epoch,
        stage1_history=pretrained.history,
        stage2_history=finetune.history,
        group_scale=finetune.best_group_scale,
        item_bias=finetune.best_item_bias,
    )
end

# -------------------------------------------------------------------------
# 10. Final five-seed experiment
# ------------------------------------------------------------------------

# Summarize multi-seed results as mean and sample standard deviation.
function summarize_results(results)
    fields = [:RMSE, :MAE, :HR5, :HR10, :HR20, :NDCG5, :NDCG10, :NDCG20]

    println("FINAL MULTI-SEED RESULTS")

    for field in fields
        values = [getproperty(r, field) for r in results]
        @printf("%-7s = %.6f +/- %.6f\n", String(field), mean(values), std(values))
    end

    println("\nCheckpoint epochs:")
    for r in results
        @printf(
            "Seed %d | Stage 1 epoch %d | Stage 2 epoch %d | Val RMSE %.6f | Val NDCG@10 %.6f\n",
            r.seed,
            r.pretrain_epoch,
            r.best_ranking_epoch,
            r.best_val_rmse,
            r.best_val_ndcg10,
        )
    end

    return nothing
end

# Run the complete experiment over multiple random seeds.
function run_five_seeds(
    data_dir;
    seeds=42:46,
    config=FINAL_CONFIG,
    verbose=false,
    gamma=0.01,
    eta=0.90,
    mu=1.0,
)
    results = NamedTuple[]

    for seed in seeds
        println("RUNNING SEED $seed")

        result = run_single_seed(
            data_dir;
            model_seed=seed,
            config=config,
            verbose=verbose,
            gamma=gamma,
            eta=eta,
            mu=mu,
        )

        push!(results, result)
    end

    summarize_results(results)
    return results
end

end # module SharedLowRankGRS
