extern crate minimax_strategy;
extern crate num;

use minimax_strategy::*;
use num::Bounded;
use std::fmt;
use std::ops::Neg;

const FIELD_SIZE: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GameResult {
    Win(Actor),
    Draw,
}

#[derive(Clone)]
struct Board {
    occupancies: [[Option<Actor>; FIELD_SIZE]; FIELD_SIZE],
}

impl Board {
    fn new() -> Self {
        Self {
            occupancies: [[None; FIELD_SIZE]; FIELD_SIZE],
        }
    }

    fn at(&self, x: usize, y: usize) -> &Option<Actor> {
        &self.occupancies[x][y]
    }

    fn at_mut(&mut self, x: usize, y: usize) -> &mut Option<Actor> {
        &mut self.occupancies[x][y]
    }

    fn game_result(&self) -> Option<GameResult> {
        // 横方向に揃っているところがあるか
        'row_check: for row in 0..FIELD_SIZE {
            let start = self.at(0, row);
            match start {
                Some(actor) => {
                    for column in 0..FIELD_SIZE {
                        if self.at(column, row) != start {
                            continue 'row_check;
                        }
                    }
                    return Some(GameResult::Win(*actor));
                }
                None => continue,
            }
        }
        // 縦方向に揃っているところがあるか
        'column_check: for column in 0..FIELD_SIZE {
            let start = self.at(column, 0);
            match start {
                Some(actor) => {
                    for row in 0..FIELD_SIZE {
                        if self.at(column, row) != start {
                            continue 'column_check;
                        }
                    }
                    return Some(GameResult::Win(*actor));
                }
                None => continue,
            }
        }
        {
            // 対角方向に揃っているところがあるか
            let start = self.at(0, 0);
            if let Some(actor) = start {
                if (0..FIELD_SIZE).map(|i| self.at(i, i)).all(|a| a == start) {
                    return Some(GameResult::Win(*actor));
                }
            }
        }
        {
            let start = self.at(0, FIELD_SIZE - 1);
            if let Some(actor) = start {
                if (0..FIELD_SIZE)
                    .map(|i| self.at(i, FIELD_SIZE - i - 1))
                    .all(|a| a == start)
                {
                    return Some(GameResult::Win(*actor));
                }
            }
        }
        // まだ何も置かれていないマスがあれば，決着はついていない
        for row in 0..FIELD_SIZE {
            for column in 0..FIELD_SIZE {
                if self.at(column, row).is_none() {
                    return None;
                }
            }
        }
        // すべてのマスが埋まっているが，揃っているものがないので，勝負は引き分け
        Some(GameResult::Draw)
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "A: Agent reversi   O: Your reversi   -: Empty mass\n")?;
        writeln!(f, "--------------------> x")?;
        writeln!(f, "|")?;
        writeln!(f, "|")?;
        writeln!(f, "V y")?;
        writeln!(f, "")?;

        for row in 0..FIELD_SIZE {
            for column in 0..FIELD_SIZE {
                let displayed_item = match self.at(column, row) {
                    Some(Actor::Agent) => "A",
                    Some(Actor::Other) => "O",
                    None => "-",
                };
                write!(f, "{} ", displayed_item)?;
            }
            writeln!(f, "")?;
        }
        Ok(())
    }
}

impl State for Board {
    fn is_game_over(&self) -> bool {
        self.game_result().is_some()
    }
}

struct Placement {
    x: usize,
    y: usize,
    actor: Actor,
}

impl Placement {
    fn new(x: usize, y: usize, actor: Actor) -> Self {
        Self { x, y, actor }
    }
}

impl fmt::Display for Placement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Player {:?} placed at ({}, {}).",
            self.actor, self.x, self.y
        )
    }
}

impl Action for Placement {
    fn actor(&self) -> Actor {
        self.actor
    }
}

struct ReversiRule {}

impl ReversiRule {
    fn new() -> Self {
        Self {}
    }
}

impl Rule<Board, Placement> for ReversiRule {
    type ActionIterator = std::vec::IntoIter<Placement>;

    fn iterate_available_actions(&self, state: &Board, actor: Actor) -> Self::ActionIterator {
        let mut actions = vec![];
        for row in 0..FIELD_SIZE {
            for column in 0..FIELD_SIZE {
                if let None = state.at(column, row) {
                    let action = Placement::new(column, row, actor);
                    actions.push(action);
                }
            }
        }
        actions.into_iter()
    }

    fn translate_state(&self, state: &Board, action: &Placement) -> Board {
        debug_assert!(state.at(action.x, action.y).is_none());
        let mut next_state = state.clone();
        *next_state.at_mut(action.x, action.y) = Some(action.actor);
        next_state
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum BoardEvaluation {
    Lose,
    OccupiedCenterMass,
    Equal,
    OccupyCenterMass,
    Win,
}

impl Bounded for BoardEvaluation {
    fn min_value() -> Self {
        BoardEvaluation::Lose
    }

    fn max_value() -> Self {
        BoardEvaluation::Win
    }
}

impl Neg for BoardEvaluation {
    type Output = Self;

    fn neg(self) -> Self {
        match self {
            BoardEvaluation::Lose => BoardEvaluation::Win,
            BoardEvaluation::OccupiedCenterMass => BoardEvaluation::OccupyCenterMass,
            BoardEvaluation::Equal => BoardEvaluation::Equal,
            BoardEvaluation::OccupyCenterMass => BoardEvaluation::OccupiedCenterMass,
            BoardEvaluation::Win => BoardEvaluation::Lose,
        }
    }
}

struct BoardEvaluator;

impl Evaluator<Board> for BoardEvaluator {
    type Evaluation = BoardEvaluation;
    fn evaluate_for_agent(&self, state: &Board) -> Self::Evaluation {
        match state.game_result() {
            Some(GameResult::Win(Actor::Agent)) => BoardEvaluation::Win,
            Some(GameResult::Win(Actor::Other)) => BoardEvaluation::Lose,
            _ => match state.at(FIELD_SIZE / 2, FIELD_SIZE / 2) {
                Some(Actor::Agent) => BoardEvaluation::OccupyCenterMass,
                Some(Actor::Other) => BoardEvaluation::OccupiedCenterMass,
                None => BoardEvaluation::Equal,
            },
        }
    }
}

fn input_index() -> usize {
    loop {
        let mut buffer = String::new();
        std::io::stdin().read_line(&mut buffer).unwrap();
        if let Ok(index) = buffer.trim().parse() {
            return index;
        }
        println!("Invalid input");
    }
}

fn input_user_action(board: &Board, reversi_rule: &ReversiRule) -> Placement {
    debug_assert!(!board.is_game_over());
    loop {
        println!("Input placement x");
        let x = input_index();
        println!("Input placement y");
        let y = input_index();
        if let Some(placement) = reversi_rule
            .iterate_available_actions(&board, Actor::Other)
            .into_iter()
            .find(|placement| placement.x == x && placement.y == y)
        {
            return placement;
        }
        println!("Invalid placement position");
    }
}

fn main() {
    let consideration_depth = FIELD_SIZE * FIELD_SIZE;
    let reversi_rule = ReversiRule::new();
    let agent_strategy = NegaAlphaStrategy::new(&reversi_rule, BoardEvaluator);
    let mut board = Board::new();
    let mut current_actor = Actor::Other;

    while !board.is_game_over() {
        println!("{}", board);
        match current_actor {
            Actor::Agent => {
                println!("Searching actions..");
                match agent_strategy.search_action(&board, consideration_depth) {
                    Some(action) => board = reversi_rule.translate_state(&board, &action),
                    None => println!("There is no available action for the agent."),
                }
                println!("Find an action");
            }
            Actor::Other => {
                board =
                    reversi_rule.translate_state(&board, &input_user_action(&board, &reversi_rule))
            }
        }
        current_actor = current_actor.the_other();
    }

    println!("{}", board);
    println!("The result is {:?}", board.game_result().unwrap());
}
