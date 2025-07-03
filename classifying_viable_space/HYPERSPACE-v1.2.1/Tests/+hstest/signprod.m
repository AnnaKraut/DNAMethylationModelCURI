function [ cost ] = signprod( v )
    cost = prod(sign(v));
end
