/// 単方向の木構造におけるノードを表す．
#[derive(Debug)]
pub(crate) struct TreeNode<T> {
    /// このノードが保持する情報．
    item: T,
    /// 任意の数の子ノード．
    children: Vec<Self>,
}

impl<T> TreeNode<T> {
    /// 子を持たないノードを作成する．
    pub fn new(item: T) -> Self {
        Self {
            item,
            children: vec![],
        }
    }

    /// このノードが保持する情報を返す．
    pub fn into_inner(self) -> T {
        self.item
    }

    /// このノードが保持する情報を返す．
    pub fn item(&self) -> &T {
        &self.item
    }

    /// このノードが保持する情報を返す．
    pub fn item_mut(&mut self) -> &mut T {
        &mut self.item
    }

    pub fn into_children(self) -> impl IntoIterator<Item = Self> {
        self.children.into_iter()
    }

    /// このノードの子を列挙する．
    pub fn children_mut(&mut self) -> impl Iterator<Item = &mut Self> {
        self.children.iter_mut()
    }

    /// このノードに指定したオブジェクトを保持する子ノードを追加する．
    pub fn add_child(&mut self, child_item: T) {
        let child = Self::new(child_item);
        self.children.push(child);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_inner() {
        let node = TreeNode::new(vec![0, 1, 2]);
        assert_eq!(vec![0, 1, 2], node.into_inner());
    }

    #[test]
    fn test_item() {
        let node = TreeNode::new(vec![0, 1, 2]);
        assert_eq!(&vec![0, 1, 2], node.item());
    }

    #[test]
    fn test_item_mut() {
        let mut node = TreeNode::new(vec![0, 1, 2]);
        node.item_mut().push(3);
        assert_eq!(&vec![0, 1, 2, 3], node.item());
    }

    #[test]
    fn test_add_child() {
        let mut node = TreeNode::new("root");
        node.add_child("child");
        assert_eq!(1, node.children_mut().count());
        assert_eq!(Some("child"), node.children_mut().nth(0).map(|c| *c.item()));
    }
}
