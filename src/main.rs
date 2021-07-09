#![allow(dead_code)]

use std::ops::Index;

fn main() {
    println!("Hello, world!");
}

pub trait Tree<I: Item> {
    fn build(items: &[I], dim: usize) -> Self;
}

pub trait Item {
    fn index(&self, i: usize) -> usize;
    fn len() -> usize;
}

struct RTreeNode<T: Item, S: Tree<T>> {
    left: Box<Self>,
    elem: T,
    sub_tree: Option<S>,
    right: Box<Self>,
}

pub struct RTree<I: Item, S: Tree<I>> {
    head: RTreeNode<I, S>,
}

// RTreeNode<Point, TreapNode>::new(&[])

impl<I: Item, S: Tree<I>> Tree<I> for RTreeNode<I, S> {
    fn build(items: &[I], dim: usize) -> Self {
        let _ = items;
        let _ = dim;
        todo!()
    }
}

struct Point {
    x: usize,
    y: usize,
}

impl Item for Point {
    fn index(&self, i: usize) -> usize {
        match i {
            0 => self.x,
            1 => self.y,
            _ => panic!("not that big of a dimension"),
        }
    }

    fn len() -> usize {
        2
    }
}
