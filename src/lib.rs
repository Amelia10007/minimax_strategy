mod node;

use node::TreeNode;
use num::{Bounded, Integer};
use std::marker::PhantomData;
use std::ops::Neg;

/// 2人ゲームにおける手番．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    type Evaluation: Copy + Ord + Bounded + Neg<Output = Self::Evaluation>;

    /// 指定された状態について，エージェントにとっての有利度合いを評価する．
    fn evaluate_for_agent(&self, state: &S) -> Self::Evaluation;
}

/// 2人完全情報ゲームの手をネガアルファ法で思考するエージェント．
pub struct NegaAlphaStrategy<'r, S, A, R, E> {
    /// ゲームルール．
    rule: &'r R,
    /// 評価関数．
    evaluator: E,
    /// 👻👻👻
    _s: PhantomData<S>,
    /// 👻👻👻
    _a: PhantomData<A>,
}

/// ミニマックス法で利用するゲーム木のノード．
#[derive(Debug)]
struct MinimaxNode<S, A, E> {
    /// 現在の状態．
    state: S,
    /// この状態に至る際に実行された行動．
    cause_action: A,
    /// エージェントにとっての現在状態の評価値．
    evaluation: E,
}

impl Actor {
    pub fn opponent(&self) -> Self {
        match self {
            Actor::Agent => Actor::Other,
            Actor::Other => Actor::Agent,
        }
    }
}

impl<'r, S, A, R, E> NegaAlphaStrategy<'r, S, A, R, E>
where
    S: State,
    A: Action,
    R: Rule<S, A>,
    E: Evaluator<S>,
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

    /// ネガアルファ法により，現在の状態に対するエージェントの望ましい行動を探索する．
    /// # Params
    /// 1. state 現在の状態
    /// 1. search_depth 何手先まで読むか．例えば，次のエージェントの手までのみ読むなら，`search_depth`は1にすれば良い．
    ///
    /// # Returns
    /// 1種類以上の行動が可能な場合，その中の最も望ましい行動`action`を`Some(action)`として返す．
    ///
    /// 可能な行動がない場合，`None`を返す．
    ///
    /// # Panics
    /// 評価値`e`の正負反転`-e`がオーバーフローした場合．
    pub fn search_action<N: Copy + Integer>(&self, state: &S, search_depth: N) -> Option<A> {
        self.rule
            // 現在の状態に対して，このエージェントができる行動をまず列挙
            .iterate_available_actions(state, Actor::Agent)
            .into_iter()
            // 試しに行動後のノードを作ってみる
            .map(|agent_action| {
                let next_state = self.rule.translate_state(state, &agent_action);
                TreeNode::new(MinimaxNode::new(
                    next_state,
                    agent_action,
                    self.evaluator.evaluate_for_agent(state),
                ))
            })
            // さらにその後の手をネガアルファ法により読む．
            // 最終的に，その後の手のミニマックス評価値がこのノードの評価値となる．
            .map(|mut root| {
                let alpha = E::Evaluation::min_value();
                let beta = E::Evaluation::max_value();
                // ネガアルファ法では，手番が変わるたびに評価関数の符号を反転させることで，自他の手を統合して思考する．
                let root_evaluation =
                    -self.alpha_beta(search_depth - N::one(), &mut root, -beta, -alpha);
                (root, root_evaluation)
            })
            // もっとも評価値が良い行動を選択する．
            .max_by(
                |(_left_root, left_evaluation), (_right_root, right_evaluation)| {
                    left_evaluation.cmp(right_evaluation)
                },
            )
            .map(|(root, _evaluation)| root.into_inner().cause_action)
    }

    fn alpha_beta<N: Copy + Integer>(
        &self,
        remaining_depth: N,
        current_node: &mut TreeNode<MinimaxNode<S, A, E::Evaluation>>,
        alpha: E::Evaluation,
        beta: E::Evaluation,
    ) -> E::Evaluation {
        use std::cmp::max;

        // 注目ノードが末端ノードなら，注目ノードの状態に対する評価値を返す．
        if remaining_depth.is_zero() || current_node.item().state.is_game_over() {
            return current_node.item().evaluation;
        }
        // who WILL act on the current state?
        let next_actor = current_node.item().cause_action.actor().opponent();
        // 次の実現しうる状態をすべて列挙し，それらを現在のノードの子に加える．
        for action in self
            .rule
            .iterate_available_actions(&current_node.item().state, next_actor)
        {
            let minimax_node = {
                let next_state = self
                    .rule
                    .translate_state(&current_node.item().state, &action);
                // ネガアルファ法では，手番によって評価値の正負を反転させる必要がある．
                let evaluation = match next_actor {
                    Actor::Agent => -self.evaluator.evaluate_for_agent(&next_state),
                    Actor::Other => self.evaluator.evaluate_for_agent(&next_state),
                };
                MinimaxNode::new(next_state, action, evaluation)
            };
            current_node.add_child(minimax_node);
        }

        // 子ノードの評価値を再帰的に求める．
        let mut alpha = alpha;
        for child in current_node.children_mut() {
            let next_depth = remaining_depth - N::one();
            // ネガアルファ法では，手番が変わるたびに評価関数の符号を反転させることで，自他の手を統合して思考する．
            alpha = max(alpha, -self.alpha_beta(next_depth, child, -beta, -alpha));
            // αカット
            if alpha >= beta {
                break;
            }
        }
        // 子ノードたちの最終的な評価値をこのノードに反映
        current_node.item_mut().evaluation = alpha;
        //
        alpha
    }
}

impl<S, A, E> MinimaxNode<S, A, E> {
    fn new(state: S, cause_action: A, evaluation: E) -> Self {
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
