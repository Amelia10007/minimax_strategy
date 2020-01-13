use std::ops::{Deref, DerefMut};

/// 子をひとつ以下持つノードを表す．
#[derive(Debug)]
pub(crate) struct TreeNode<T> {
    /// このノードが保持する情報．
    item: T,
    /// 子ノード．
    child: Option<Box<Self>>,
}

impl<T> TreeNode<T> {
    /// 子を持たないノードを作成する．
    pub const fn new(item: T) -> Self {
        Self { item, child: None }
    }

    /// このノードが保持する情報を返す．
    pub fn into_inner(self) -> T {
        self.item
    }

    /// このノードの子ノードが存在すれば，それを返す．
    pub fn into_child(self) -> Option<Self> {
        self.child.map(|c| *c)
    }

    /// このノードの子ノードを，指定したノードに置き換える．
    /// この処理の前にすでに子ノードが存在していた場合，その子は破棄される．
    pub fn replace_child(&mut self, new_child: Self) {
        self.child = Some(Box::new(new_child));
    }
}

impl<T> Deref for TreeNode<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

impl<T> DerefMut for TreeNode<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.item
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
        assert_eq!(&vec![0, 1, 2], node.deref());
    }

    #[test]
    fn test_item_mut() {
        let mut node = TreeNode::new(vec![0, 1, 2]);
        node.push(3);
        assert_eq!(&vec![0, 1, 2, 3], node.deref());
    }

    #[test]
    fn test_replace_child() {
        let node = TreeNode::new("root");
        assert_eq!(None, node.into_child().map(|c| *c.deref()));

        let mut node = TreeNode::new("root");
        node.replace_child(TreeNode::new("child"));
        assert_eq!(Some("child"), node.into_child().map(|c| *c.deref()));

        let mut node = TreeNode::new("root");
        node.replace_child(TreeNode::new("child1"));
        node.replace_child(TreeNode::new("child2"));
        assert_eq!(Some("child2"), node.into_child().map(|c| *c.deref()));
    }
}
