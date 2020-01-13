mod node;

use node::TreeNode;
use num::{Bounded, Integer};
use std::marker::PhantomData;

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
    S: State,
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
        self.alpha_beta(
            search_depth,
            &mut root,
            E::Evaluation::min_value(),
            E::Evaluation::max_value(),
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
    fn alpha_beta<N: Copy + Integer>(
        &self,
        remaining_depth: N,
        current_node: &mut TreeNode<MinimaxNode<S, A, E::Evaluation>>,
        alpha: E::Evaluation,
        beta: E::Evaluation,
    ) -> E::Evaluation {
        // デバッグ用アサーション (消しても問題ないけど，コード変更した際の挙動検証のために一応とっておく)
        debug_assert!(alpha <= beta);
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
        // 次の実現しうる状態をすべて列挙
        let realizable_children = self
            .rule
            .iterate_available_actions(&current_node.state, next_actor)
            .into_iter()
            .map(|action| {
                let next_state = self.rule.translate_state(&current_node.state, &action);
                MinimaxNode::new(next_state, Some(action), None)
            })
            .map(|minimax_node| TreeNode::new(minimax_node))
            .collect::<Vec<_>>();

        // 行動がない場合は，現在の状態に対する静的評価値をそのまま適用する
        if realizable_children.is_empty() {
            let evaluation = self.evaluator.evaluate_for_agent(&current_node.state);
            current_node.evaluation = Some(evaluation);
            return evaluation;
        }

        // 注目ノードの評価値を，子ノードの評価値を用いて再帰的に求める．
        let next_depth = remaining_depth - N::one();
        match next_actor {
            // 子ノードがエージェントの行動によって実現される場合
            // エージェントは自分が有利になるよう意思決定するので，子ノードの中から評価値が最も高いものを選ぶ
            Actor::Agent => {
                let mut alpha = alpha;
                for mut child in realizable_children.into_iter() {
                    let child_evaluation = self.alpha_beta(next_depth, &mut child, alpha, beta);
                    // より評価値が高い子が見つかれば，そのノードを注目ノードの子として登録する
                    match current_node.evaluation {
                        Some(e) if e >= child_evaluation => continue,
                        _ => {}
                    }
                    current_node.evaluation = Some(child_evaluation);
                    alpha = child_evaluation;
                    current_node.replace_child(child);
                    // βカット
                    if alpha >= beta {
                        break;
                    }
                }
            }
            // 子ノードが敵の行動によって実現される場合
            // 敵はエージェントが不利になるよう意思決定するはずなので，子ノードの中から評価値が最も低いものを選ぶ
            Actor::Other => {
                let mut beta = beta;
                for mut child in realizable_children.into_iter() {
                    let child_evaluation = self.alpha_beta(next_depth, &mut child, alpha, beta);
                    // より評価値が低い子が見つかれば，そのノードを注目ノードの子として登録する
                    match current_node.evaluation {
                        Some(e) if e <= child_evaluation => continue,
                        _ => {}
                    }
                    current_node.evaluation = Some(child_evaluation);
                    beta = child_evaluation;
                    current_node.replace_child(child);
                    // αカット
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        // ここに到達する時点で，注目ノードには1つ以上の子ノードが存在するので，その子ノードの評価値が注目ノードの評価値に反映されているはずである．
        // つまり，注目ノードの評価値が確定しているので，このunwrap()は必ず成功する．
        current_node.evaluation.unwrap()
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
