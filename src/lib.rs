mod node;

use data_structure::Range;
use node::TreeNode;
use num::{Bounded, Integer};
use std::marker::PhantomData;

/// 2人ゲームにおける手番．
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Actor {
    /// 思考エージェント．
    Agent,
    /// ユーザなどの他エージェント．
    Other,
}

/// ゲームの状態．
pub trait State {
    /// この状態がすでにゲーム終了条件を満たしているか．
    fn is_game_over(&self) -> bool;
}

/// ゲームにおける行動．
pub trait Action {
    /// この行動の手番．
    fn actor(&self) -> Actor;
}

/// ゲーム内の状態遷移条件を記述する．
pub trait Rule<S, A> {
    /// ある状態において実行可能な行動を列挙する際に使用する型．
    type ActionIterator: IntoIterator<Item = A>;

    /// 指定された状態下で実行可能な行動を列挙する．
    fn iterate_available_actions(&self, state: &S, actor: Actor) -> Self::ActionIterator;

    /// 状態を遷移させる．
    fn translate_state(&self, state: &S, action: &A) -> S;
}

/// ゲーム状態の評価関数．
pub trait Evaluator<S> {
    /// 評価指標となる型．
    type Evaluation;

    /// 指定された状態について，エージェントにとっての有利度合いを評価する．
    fn evaluate_for_agent(&self, state: &S) -> Self::Evaluation;
}

/// 2人零和ゲームにおける適切な行動をαβ法で思考するエージェント．
pub struct AlphaBetaStrategy<'r, S, A, R, E> {
    /// ゲームルール．
    rule: &'r R,
    /// 評価関数．
    evaluator: E,
    /// 👻
    _s: PhantomData<S>,
    /// 👻
    _a: PhantomData<A>,
}

/// ミニマックス法で利用するゲーム木のノード．
#[derive(Debug)]
struct MinimaxNode<S, A, E> {
    /// 現在の状態．
    state: S,
    /// この状態に至る際に実行された行動．
    cause_action: Option<A>,
    /// エージェントにとっての現在状態の評価値．
    evaluation: Option<E>,
}

impl Actor {
    /// この手番に対する相手側の手番を返す．
    pub fn opponent(&self) -> Self {
        match self {
            Actor::Agent => Actor::Other,
            Actor::Other => Actor::Agent,
        }
    }
}

impl<'r, S, A, R, E> AlphaBetaStrategy<'r, S, A, R, E>
where
    S: State + Clone,
    A: Action,
    R: Rule<S, A>,
    E: Evaluator<S>,
    E::Evaluation: Copy + Ord + Bounded,
{
    /// 指定したゲームルールおよび評価関数のもと思考するエージェントを生成する．
    pub fn new(rule: &'r R, evaluator: E) -> Self {
        Self {
            rule,
            evaluator,
            _s: PhantomData,
            _a: PhantomData,
        }
    }

    /// αβ法により，現在の状態に対するこのエージェントの望ましい行動を探索する．
    /// # Params
    /// 1. state 現在の状態
    /// 1. search_depth 何手先まで読むか．例えば，次のエージェントの手までのみ読むなら，`search_depth`は1にすれば良い．
    ///
    /// # Returns
    /// 1種類以上の行動が可能な場合，その中の最も望ましい行動`a`とその行動後の状態`s`を`Some((a, s))`として返す．
    ///
    /// 可能な行動がない場合，`None`を返す．
    pub fn search_action<N: Copy + Integer>(&self, state: S, search_depth: N) -> Option<(A, S)> {
        let mut root = TreeNode::new(MinimaxNode::new(state, None, None));
        self.construct_best_game_tree_alpha_beta(
            search_depth,
            &mut root,
            Range::new(E::Evaluation::min_value(), E::Evaluation::max_value()),
        );
        root.into_child().and_then(|best_node| {
            let inner = best_node.into_inner();
            let next_state = inner.state;
            inner.cause_action.map(|action| (action, next_state))
        })
    }

    /// αβ法により，指定したノードの評価値を再帰的に計算する．
    /// # Params
    /// 1. remaining_depth 残りの探索深さ．
    /// 1. current_node 注目ノード．
    /// 1. alpha 評価値の関心範囲の下限．
    /// 1. beta 評価値の関心範囲の上限．
    fn construct_best_game_tree_alpha_beta<N: Copy + Integer>(
        &self,
        remaining_depth: N,
        current_node: &mut TreeNode<MinimaxNode<S, A, E::Evaluation>>,
        evaluation_range: Range<E::Evaluation>,
    ) -> E::Evaluation {
        // デバッグ用アサーション (消しても問題ないけど，コード変更した際の挙動検証のために一応とっておく)
        debug_assert!(current_node.evaluation.is_none());

        // 注目ノードが末端ノードなら，現在の状態に対する静的評価値をそのまま適用する
        if remaining_depth.is_zero() || current_node.state.is_game_over() {
            let evaluation = self.evaluator.evaluate_for_agent(&current_node.state);
            current_node.evaluation = Some(evaluation);
            return evaluation;
        }

        // who WILL act on the current state?
        let next_actor = match current_node.cause_action.as_ref() {
            Some(action) => action.actor().opponent(),
            None => Actor::Agent,
        };

        // 注目ノードの評価値を，子ノードの評価値を用いて再帰的に求める．
        let current_state = current_node.state.clone();
        let mut current_evaluation_range = evaluation_range;

        // 次の実現しうる状態をすべて列挙し，ひとつひとつ調べる
        for mut child in self
            .rule
            .iterate_available_actions(&current_state, next_actor)
            .into_iter()
            .map(|action| {
                let next_state = self.rule.translate_state(&current_state, &action);
                MinimaxNode::new(next_state, Some(action), None)
            })
            .map(|minimax_node| TreeNode::new(minimax_node))
        {
            let child_evaluation = self.construct_best_game_tree_alpha_beta(
                remaining_depth - N::one(),
                &mut child,
                current_evaluation_range,
            );
            // ミニマックス法により，探索する必要がある枝だけを選択する
            if let Some(e) = current_node.evaluation {
                match next_actor {
                    // エージェントは自分が有利になる行動を選択するので，
                    // 自分が不利になる行動は候補から除外する
                    Actor::Agent => {
                        if e >= child_evaluation {
                            continue;
                        }
                    }
                    // 相手はエージェントが不利になる行動を選択するので，
                    // エージェントが有利になる行動は候補から除外する．
                    Actor::Other => {
                        if e <= child_evaluation {
                            continue;
                        }
                    }
                }
            }
            //
            current_node.evaluation = Some(child_evaluation);
            current_node.replace_child(child);
            // 評価値の注目範囲を更新する．
            // 可能なら，αβカットして探索量を減らす．
            let maybe_next_range = match next_actor {
                Actor::Agent => Range::try_new(child_evaluation, current_evaluation_range.max),
                Actor::Other => Range::try_new(current_evaluation_range.min, child_evaluation),
            };
            match maybe_next_range {
                Some(range) => current_evaluation_range = range,
                None => break,
            }
        }

        // ここに到達する時点で，注目ノードには1つ以上の子ノードが存在するので，その子ノードの評価値が注目ノードの評価値に反映されているはずである．
        // 行動がない場合は，現在の状態に対する静的評価値をそのまま適用する
        match current_node.evaluation {
            Some(e) => e,
            None => {
                let evaluation = self.evaluator.evaluate_for_agent(&current_state);
                current_node.evaluation = Some(evaluation);
                evaluation
            }
        }
    }
}

impl<S, A, E> MinimaxNode<S, A, E> {
    fn new(state: S, cause_action: Option<A>, evaluation: Option<E>) -> Self {
        Self {
            state,
            cause_action,
            evaluation,
        }
    }
}

#[cfg(test)]
mod test_cmp {
    #[test]
    fn test_cmp() {
        let v = vec![5, 2, 0, 9, 4, 3];
        let max = v.into_iter().max_by(|left, right| left.cmp(right));
        assert_eq!(Some(9), max);
    }
}
