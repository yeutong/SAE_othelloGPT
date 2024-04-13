import os
import numpy as np
import torch
import torch.nn as nn
from torcheval.metrics import BinaryAUROC

from torch import Tensor
from jaxtyping import Float, Int

import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from scipy.stats import skew


from analysis import get_autoencoder_directions, plot_board, show_top_activating

all_activations = torch.load("analysis_results/all_activations.pkl")
all_boards = torch.load("analysis_results/all_boards.pkl")
all_legal_moves = torch.load("analysis_results/all_legal_moves_me.pkl")


def get_probe_directions(
    probe_location="probes/probe_layer_6.pkl", normalized=True
) -> Float[Tensor, "pos cls d_model"]:

    with open(probe_location, "rb") as f:
        probe = torch.load(f, map_location=device)

    original_probe_directions = torch.stack([x.weight for x in probe.classifier], dim=1)
    if normalized:
        probe_directions: Float[Tensor, "pos cls d_model"] = (
            original_probe_directions
            / original_probe_directions.norm(dim=-1, keepdim=True)
        )
        return probe_directions
    return original_probe_directions


def get_probe_activations(
    act_location="analysis_results/probe_activations.pkl",
) -> Float[Tensor, "n_board pos cls"]:
    with open(act_location, "rb") as f:
        probe_activations = torch.load(f)
    return probe_activations


def get_probe_boards(
    board_location="analysis_results/probe_all_boards.pkl",
) -> Float[Tensor, "n_board pos"]:
    with open(board_location, "rb") as f:
        probe_boards = torch.load(f)
    return probe_boards


def pos_string2int(pos_name: str) -> Int:
    """Example: A1 -> 0, F6 -> 45, H8 -> 63"""
    assert len(pos_name) == 2, "pos_name must be a string of length 2"
    letter, number = pos_name
    return (ord(letter) - ord("A")) + (int(number) - 1) * 8


def pos_int2string(pos_int: Int) -> str:
    """Example: 0 -> A1, 45 -> F6, 63 -> H8"""
    assert 0 <= pos_int < 64, "pos_int must be an int between 0 and 63"
    letter = chr(ord("A") + pos_int % 8)
    number = (pos_int // 8) + 1
    return f"{letter}{number}"


def get_probe_direction(
    board_position, primary_class, normalized=False
) -> Float[Tensor, "d_model"]:
    """get one probe direction for a given board position and primary class"""

    # convert string to int
    if isinstance(board_position, str):
        board_position = pos_string2int(board_position)
        assert (
            0 <= board_position < 64
        ), "Invalid board position input. Example: 'A1', 'F6'"
    elif isinstance(board_position, int):
        assert (
            0 <= board_position < 64
        ), "board_position must be an int between 0 and 63"
    else:
        raise ValueError("board_position must be a string or an int")

    # convert primary class to int
    if isinstance(primary_class, str):
        assert primary_class in [
            "Empty",
            "Own",
            "Enemy",
        ], "primary_class must be one of 'Empty', 'Own', 'Enemy'"
        primary_class = {"Empty": 0, "Own": 1, "Enemy": 2}[primary_class]
    elif isinstance(primary_class, int):
        assert primary_class in [0, 1, 2], "primary_class must be one of 0, 1, 2"
    else:
        raise ValueError("primary_class must be a string or an int")

    # check global variable for probe directions
    if normalized:
        if "probe_directions_normalized" not in globals():
            global probe_directions_normalized
            probe_directions_normalized = get_probe_directions(normalized=normalized)
        return probe_directions_normalized[board_position, primary_class]

    if "probe_directions" not in globals():
        global probe_directions
        probe_directions = get_probe_directions(normalized=normalized)
    return probe_directions[board_position, primary_class]


def show_top_activating_for_probe(
    probe_number_64,
    probe_number_3,
    top_k=5,
    marked_position=-1,
    directory="analysis_results",
):
    with open("analysis_results/probe_activations.pkl", "rb") as f:
        all_activations = torch.load(f)
    with open("analysis_results/probe_all_boards.pkl", "rb") as f:
        all_boards = torch.load(f)
    num_data = all_boards.shape[0]
    filtered_boards = all_boards[torch.where(all_boards[:, 0] > -100)]
    these_activations = all_activations[:, probe_number_64, probe_number_3]
    filtered_activations = these_activations[torch.where(all_boards[:, 0] > -100)]
    best_activations, best_indices = torch.topk(filtered_activations, k=top_k)
    random_indices = torch.randint(low=0, high=num_data, size=(top_k,))
    best_boards = filtered_boards[best_indices]
    # best_activations=sorted_activations[best_indices]
    random_boards = filtered_boards[random_indices]
    random_activations = filtered_activations[random_indices]
    class_names = {0: "Empty", 1: "Own", 2: "Enemy"}

    fig, axs = plt.subplots(2, top_k, figsize=(10, 5))
    plt.suptitle(
        f"Top Activating Boards and Random Boards for Probe {probe_number_64}-{class_names[probe_number_3]}, with Position {marked_position} marked"
    )

    for n in range(top_k):
        # Plot on the specified axes in the grid
        plot_board(
            axs[0, n],
            best_boards[n],
            best_activations[n],
            marked_position=marked_position,
        )
        plot_board(
            axs[1, n],
            random_boards[n],
            random_activations[n],
            marked_position=marked_position,
        )

    plt.show()
    plt.close(fig)  # Close the specific figure to release memory


def gen_pattern(
    empty: list[str] = ["D6"],
    enemy: list[str] = ["E6"],
    own: list[str] = ["F6"],
):
    """
    generate a pattern for a given board position
    """
    # initialize the board to -1
    pattern = torch.full((64,), -1)
    for pos in empty:
        pattern[pos_string2int(pos)] = 0
    for pos in own:
        pattern[pos_string2int(pos)] = 1
    for pos in enemy:
        pattern[pos_string2int(pos)] = 2
    return pattern


def gen_pattern_by_string(
    based_token: str = "D6",
    direction: str = "right",
    num_of_enemy: int = 1,
):
    """
    generate a pattern for a given board position
    """
    # initialize the board to -1
    assert direction in [
        "right",
        "left",
        "up",
        "down",
        "up_right",
        "up_left",
        "down_right",
        "down_left",
    ], "Invalid direction"

    # get the board to 2-dimensional
    based_token = pos_string2int(based_token)
    r, c = based_token // 8, based_token % 8

    board = torch.full((8, 8), -1)

    # add the empty piece
    board[r, c] = 0

    dir_vector = {
        "right": (0, 1),
        "left": (0, -1),
        "up": (-1, 0),
        "down": (1, 0),
        "up_right": (-1, 1),
        "up_left": (-1, -1),
        "down_right": (1, 1),
        "down_left": (1, -1),
    }[direction]

    # add the enemy and own pieces
    for i in range(num_of_enemy + 1):
        r += dir_vector[0]
        c += dir_vector[1]

        # check if the position is out of the board
        if r < 0 or r >= 8 or c < 0 or c >= 8:
            return None

        if i < num_of_enemy:
            # add the enemy piece
            board[r, c] = 2
        else:
            # add the own piece
            board[r, c] = 1

    return board.flatten()


def gen_pattern_dict_from_pos(pos: str):
    """generate pattern from given position"""
    directions = [
        "right",
        "left",
        "up",
        "down",
        "up_right",
        "up_left",
        "down_right",
        "down_left",
    ]
    patterns_dict = {}

    for direction in directions:
        for num_of_enemy in range(1, 8):
            pattern = gen_pattern_by_string("D6", direction, num_of_enemy)
            if pattern is None:
                break

            patterns_dict[f"{direction}_{num_of_enemy}"] = pattern

    return patterns_dict


def _plot_distribution(activations, labels, title, ignore_zero=False):
    if ignore_zero:
        activations = [act[act > 0] for act in activations]
    activations = [act.detach().numpy() for act in activations]

    # show probability density
    fig = ff.create_distplot(
        hist_data=activations,
        group_labels=labels,
        bin_size=1,
    )

    fig.update_layout(title=title, width=800, height=400)
    fig.show()


def plot_distribution_of_activations_legal(
    feature_number,
    target_position,
    ignore_zero=False,
):
    feature_activation: Float[Tensor, "n_board"] = all_activations[:, feature_number]
    board_position: Float[Tensor, "n_board"] = all_legal_moves[:, target_position]

    legal_activations = feature_activation[board_position == 1]
    illegal_activations = feature_activation[board_position == 0]

    pos_string = pos_int2string(target_position)
    _plot_distribution(
        [legal_activations, illegal_activations],
        [f"{pos_string} is Legal", f"{pos_string} is Illegal"],
        title=f"Feature {feature_number} Activations Distribution",
        ignore_zero=ignore_zero,
    )


def plot_distribution_of_activations_types(
    feature_number,
    target_position,
    ignore_zero=False,
):
    feature_activation: Float[Tensor, "n_board"] = all_activations[:, feature_number]
    board_position: Float[Tensor, "n_board"] = all_boards[:, target_position]

    empty_activations = feature_activation[board_position == 0]
    own_activations = feature_activation[board_position == 1]
    enemy_activations = feature_activation[board_position == 2]

    pos_string = pos_int2string(target_position)
    _plot_distribution(
        [empty_activations, own_activations, enemy_activations],
        [f"{pos_string} is Empty", f"{pos_string} is Own", f"{pos_string} is Enemy"],
        title=f"Feature {feature_number} Activations Distribution",
        ignore_zero=ignore_zero,
    )


def plot_distribution_of_activations_empty_legal(
    feature_number,
    target_position,
    ignore_zero=False,
):
    """
    show distribution of "empty + legal" and "empty + illegal" activations
    """
    feature_activation: Float[Tensor, "n_board"] = all_activations[:, feature_number]
    board_position: Float[Tensor, "n_board"] = all_boards[:, target_position]

    legal_activations = feature_activation[
        (board_position == 0) & (all_legal_moves[:, target_position] == 1)
    ]
    illegal_activations = feature_activation[
        (board_position == 0) & (all_legal_moves[:, target_position] == 0)
    ]

    pos_string = pos_int2string(target_position)
    _plot_distribution(
        [legal_activations, illegal_activations],
        [f"{pos_string} is Empty + Legal", f"{pos_string} is Empty + Illegal"],
        title=f"Feature {feature_number} Activations Distribution",
        ignore_zero=ignore_zero,
    )


def plot_distribution_of_activations_odd_even_empty_legal(
    feature_number,
    target_position,
    ignore_zero=False,
):
    """
    show distribution of "empty + legal" and "empty + illegal" activations
    """
    feature_activation: Float[Tensor, "n_board"] = all_activations[:, feature_number]
    board_position: Float[Tensor, "n_board"] = all_boards[:, target_position]

    # calc sum of moves, odd or even.
    # filter board value 2, change to 1
    total_moves = all_boards.clone()
    total_moves[total_moves == 2] = 1
    total_moves = torch.sum(total_moves, dim=-1)
    even_moves = total_moves % 2 == 0

    legal_odd_activations = feature_activation[
        (board_position == 0)
        & (all_legal_moves[:, target_position] == 1)
        & (even_moves == 0)
    ]
    legal_even_activations = feature_activation[
        (board_position == 0)
        & (all_legal_moves[:, target_position] == 1)
        & (even_moves == 1)
    ]
    illegal_odd_activations = feature_activation[
        (board_position == 0)
        & (all_legal_moves[:, target_position] == 0)
        & (even_moves == 0)
    ]
    illegal_even_activations = feature_activation[
        (board_position == 0)
        & (all_legal_moves[:, target_position] == 0)
        & (even_moves == 1)
    ]

    _plot_distribution(
        [
            legal_odd_activations,
            legal_even_activations,
            illegal_odd_activations,
            illegal_even_activations,
        ],
        [
            "Empty + Legal + Odd",
            "Empty + Legal + Even",
            "Empty + Illegal + Odd",
            "Empty + Illegal + Even",
        ],
        title=f"Distribution of Activations for Feature {feature_number} at Position {target_position}",
        ignore_zero=ignore_zero,
    )


def plot_distribution_of_activations_previous_legal(
    feature_number,
    target_position,
    ignore_zero=False,
):
    """
    whether it is legal in previous move
    """
    feature_activation: Float[Tensor, "n_board"] = all_activations[:, feature_number]
    board_position: Float[Tensor, "n_board"] = all_boards[:, target_position]

    # calc sum of moves, 9 - 60
    # filter board value 2, change to 1
    total_moves = all_boards.clone()
    total_moves[total_moves == 2] = 1
    total_moves = torch.sum(total_moves, dim=-1)

    # move legal in previous move
    legal_prev_moves = all_legal_moves[:, target_position].clone()  # shape: n_board
    legal_prev_moves = torch.cat((torch.tensor([0]), legal_prev_moves[:-1]))

    # if total_moves is 9, then legal_prev_moves is 0
    legal_prev_moves[total_moves == 9] = 0

    legal_previous_activations = feature_activation[
        (board_position == 0) & (legal_prev_moves == 1) & (total_moves > 9)
    ]
    illegal_previous_activations = feature_activation[
        (board_position == 0) & (legal_prev_moves == 0) & (total_moves > 9)
    ]

    _plot_distribution(
        [legal_previous_activations, illegal_previous_activations],
        ["Empty + Legal + Previous", "Empty + Illegal + Previous"],
        title=f"Distribution of Activations for Feature {feature_number} at Position {target_position}",
        ignore_zero=ignore_zero,
    )


def plot_distribution_of_activations_current_previous_legal(
    feature_number,
    target_position,
    ignore_zero=False,
):
    """
    whether it is legal in previous move, and current move
    """
    feature_activation: Float[Tensor, "n_board"] = all_activations[:, feature_number]
    board_position: Float[Tensor, "n_board"] = all_boards[:, target_position]

    # calc sum of moves, 9 - 60
    # filter board value 2, change to 1
    total_moves = all_boards.clone()
    total_moves[total_moves == 2] = 1
    total_moves = torch.sum(total_moves, dim=-1)

    # move legal in previous move
    legal_prev_moves = all_legal_moves[:, target_position].clone()  # shape: n_board
    legal_prev_moves = torch.cat((torch.tensor([0]), legal_prev_moves[:-1]))

    # instead of prev moves, get the next moves legal/illegal status
    # legal_prev_moves = torch.cat((legal_prev_moves[1:], torch.tensor([0])))

    # legal current move
    legal_curr_moves = all_legal_moves[:, target_position]

    # if total_moves is 9, then legal_prev_moves is 0
    legal_prev_moves[total_moves == 9] = 0

    legal_prev_legal_curr_act = feature_activation[
        (board_position == 0)
        & (legal_prev_moves == 1)
        & (legal_curr_moves == 1)
        & (total_moves > 9)
    ]
    illegal_prev_legal_curr_act = feature_activation[
        (board_position == 0)
        & (legal_prev_moves == 0)
        & (legal_curr_moves == 1)
        & (total_moves > 9)
    ]
    legal_prev_illegal_curr_act = feature_activation[
        (board_position == 0)
        & (legal_prev_moves == 1)
        & (legal_curr_moves == 0)
        & (total_moves > 9)
    ]
    illegal_prev_illegal_curr_act = feature_activation[
        (board_position == 0)
        & (legal_prev_moves == 0)
        & (legal_curr_moves == 0)
        & (total_moves > 9)
    ]

    _plot_distribution(
        [
            legal_prev_legal_curr_act,
            illegal_prev_legal_curr_act,
            legal_prev_illegal_curr_act,
            illegal_prev_illegal_curr_act,
        ],
        [
            "Empty+PrevLegal+CurrLegal",
            "Empty+PrevIllegal+CurrLegal",
            "Empty+PrevLegal+CurrIllegal",
            "Empty+PrevIllegal+CurrIllegal",
        ],
        title=f"Distribution of Activations for Feature {feature_number} at Position {target_position}",
        ignore_zero=ignore_zero,
    )


def plot_distribution_of_activations_pattern(
    feature_number,
    target_position,
    pattern=gen_pattern(),
    ignore_zero=False,
):
    """
    show distribution of board containing/missing a pattern
    """
    feature_activation: Float[Tensor, "n_board"] = all_activations[:, feature_number]
    board_position: Float[Tensor, "n_board"] = all_boards[:, target_position]

    # filter board that contains the pattern, ignore -1. Only consider 0, 1, 2
    # the pattern is a shappe (64,) tensor, with -1 for any position that is not considered.
    # each board is a shape (64,) tensor, with 0, 1, 2 for empty, own, enemy
    relevant_positions = pattern != -1
    expanded_pattern = pattern[relevant_positions].unsqueeze(0)

    # Check for matches only at relevant positions
    matches = all_boards[:, relevant_positions] == expanded_pattern
    match_indices = matches.all(dim=1)

    matched_activations = feature_activation[(match_indices) & (board_position == 0)]
    unmatched_activations = feature_activation[(~match_indices) & (board_position == 0)]

    _plot_distribution(
        [matched_activations, unmatched_activations],
        ["Empty + Pattern Matched", "Empty + Pattern Unmatched"],
        title=f"Distribution of Activations for Feature {feature_number} at Position {target_position}",
        ignore_zero=ignore_zero,
    )


def plot_distribution_of_activations_patterns(
    feature_number,
    target_position,
    patterns,
    ignore_zero=False,
):
    """
    show distribution of board containing any patterns or not
    """
    feature_activation: Float[Tensor, "n_board"] = all_activations[:, feature_number]
    board_position: Float[Tensor, "n_board"] = all_boards[:, target_position]
    legal_moves: Float[Tensor, "n_board"] = all_legal_moves[:, target_position]

    # filter board that contains the pattern, ignore -1. Only consider 0, 1, 2
    # the pattern is a shappe (64,) tensor, with -1 for any position that is not considered.
    # each board is a shape (64,) tensor, with 0, 1, 2 for empty, own, enemy
    # TODOOOOOOO
    #
    match_indices_each_pattern = torch.zeros(
        (len(patterns), all_boards.shape[0]), dtype=torch.bool
    )
    for pi, pattern in enumerate(patterns):
        relevant_positions = pattern != -1
        expanded_pattern = pattern[relevant_positions].unsqueeze(0)

        # Check for matches only at relevant positions
        matches = all_boards[:, relevant_positions] == expanded_pattern
        match_indices = matches.all(dim=1)
        match_indices_each_pattern[pi] = match_indices

    # union the match indices
    match_indices = match_indices_each_pattern.any(dim=0)

    matched_activations = feature_activation[(match_indices) & (board_position == 0)]
    unmatched_activations_illegal = feature_activation[
        (~match_indices) & (board_position == 0) & (legal_moves == 0)
    ]
    unmatched_activations_legal = feature_activation[
        (~match_indices) & (board_position == 0) & (legal_moves == 1)
    ]

    _plot_distribution(
        [
            matched_activations,
            unmatched_activations_illegal,
            unmatched_activations_legal,
        ],
        [
            "Empty + Pattern Matched (Legal)",
            "Empty + Pattern Unmatched (Illegal)",
            "Empty + Pattern Unmatched (Legal)",
        ],
        title=f"Distribution of Activations for Feature {feature_number} at Position {target_position}",
        ignore_zero=ignore_zero,
    )


def plot_distribution_of_activations_pattern_legal(
    feature_number,
    target_position,
    pattern=gen_pattern(),
    pattern_name=None,
    ignore_zero=False,
):
    """
    show distribution of board containing/missing a pattern
    """
    feature_activation: Float[Tensor, "n_board"] = all_activations[:, feature_number]
    board_position: Float[Tensor, "n_board"] = all_boards[:, target_position]
    legal_moves: Float[Tensor, "n_board"] = all_legal_moves[:, target_position]

    # filter board that contains the pattern, ignore -1. Only consider 0, 1, 2
    # the pattern is a shappe (64,) tensor, with -1 for any position that is not considered.
    # each board is a shape (64,) tensor, with 0, 1, 2 for empty, own, enemy
    relevant_positions = pattern != -1
    expanded_pattern = pattern[relevant_positions].unsqueeze(0)

    # Check for matches only at relevant positions
    matches = all_boards[:, relevant_positions] == expanded_pattern
    match_indices = matches.all(dim=1)

    matched_activations = feature_activation[(match_indices) & (board_position == 0)]
    unmatched_activations_illegal = feature_activation[
        (~match_indices) & (board_position == 0) & (legal_moves == 0)
    ]
    unmatched_activations_legal = feature_activation[
        (~match_indices) & (board_position == 0) & (legal_moves == 1)
    ]

    pos_string = pos_int2string(target_position)
    _plot_distribution(
        [
            matched_activations,
            unmatched_activations_illegal,
            unmatched_activations_legal,
        ],
        [
            "Empty + Pattern Matched (Legal)",
            "Empty + Pattern Unmatched (Illegal)",
            "Empty + Pattern Unmatched (Legal)",
        ],
        title=f"Feature {feature_number} Activation Distribution at Position {pos_string} with Pattern {pattern_name}",
        ignore_zero=ignore_zero,
    )
