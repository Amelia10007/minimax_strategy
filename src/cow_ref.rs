pub enum CowRef<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<'a, T> From<T> for CowRef<'a, T> {
    fn from(t: T) -> CowRef<'a, T> {
        CowRef::Owned(t)
    }
}

impl<'a, T> From<&'a T> for CowRef<'a, T> {
    fn from(t: &'a T) -> CowRef<'a, T> {
        CowRef::Borrowed(t)
    }
}

impl<'a, T> AsRef<T> for CowRef<'a, T> {
    fn as_ref(&self) -> &T {
        match self {
            CowRef::Borrowed(b) => b,
            CowRef::Owned(o) => &o,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CowRef;
    #[test]
    fn test_cor_ref_from_owned() {
        let cr = CowRef::Owned(String::from("abc"));
        assert_eq!("abc", cr.as_ref())
    }
    #[test]
    fn test_cow_ref_from_borrowed() {
        let s = &String::from("abc");
        let cr = CowRef::Borrowed(s);
        assert_eq!("abc", cr.as_ref())
    }
}
