from value import Value

class TestValue:
    def test_value_add(self):
        a = Value(5)
        b = Value(3)
        c = a+b
        
        assert c.data == a.data + b.data

    def test_value_mul(self):
        a = Value(5)
        b = Value(3)
        c = a*b
        
        assert c.data == a.data * b.data
    
    def test_value_backwards(self):
        a = Value(5)
        b = Value(3)
        c = a*b

        d = Value(2)
        e = c/d
        elems = [a,b,c,d,e]
        assert all(isinstance(elem, Value) for elem in elems)
        e.backward()

        print([elem.grad for elem in elems])

        
