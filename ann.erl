% based off of tutorial 
% http://www.trapexit.org/Erlang_and_Neural_Networks

-module(ann).
-export([perceptron/3, connect/2]).


perceptron(Weights, Inputs, Sensitivities) ->
  Sigmoid = fun(X) -> 1/(1+math:exp(-X)) end,
  Sigmoid_deriv = fun(X) -> math:exp(-X)/(1+math:exp(-2*X)) end,

  receive
    {stimulate, Input} ->
      % add input to inputs
      New_inputs = replace_input(Inputs, Input),
      
      {In_PID, In_Val} = Input,
      ann_grapher ! {stimulate, In_PID, self(), In_Val},
      % calculate perceptron output
      Output = feed_forward(Sigmoid, Weights, 
                        convert_to_values(New_inputs)),

      % stimulate connected perceptrons
      case Sensitivities of
        [] -> 
          ann_grapher ! {output, self(), Output},
          io:format("Output (~w): ~w~n", [self(), Output]),
          self() ! {learn, {self(), 1}};
        S -> lists:foreach(fun(Output_PID) ->
              Output_PID ! {stimulate, {self(), Output}}
            end,
            convert_to_keys(S))
      end,
      perceptron(Weights, New_inputs, Sensitivities);

    {learn, Backprop} ->
    
      {Back_source, Back_value} = Backprop,
      ann_grapher ! {backprop, Back_source, self(), Back_value},

      Learning_rate = 0.5,
      New_sensitivities = add_sensitivity(Sensitivities, Backprop),
      Output = feed_forward(Sigmoid, Weights, 
                  convert_to_values(Inputs)),
      Deriv = feed_forward(Sigmoid_deriv, Weights, 
                  convert_to_values(Inputs)),
      Sensitivity = calculate_sensitivities(Backprop, Inputs,
                  New_sensitivities, Output, Deriv),

      Weight_adjustments = lists:map(fun(Input) ->
                              Learning_rate * Sensitivity * Input
                            end,
                            convert_to_values(Inputs)),
      New_weights = vector_map(fun(W, D) -> W+D end, Weights,
                      Weight_adjustments),

      vector_map(fun(Weight, Input_PID) ->
          Input_PID ! {learn, {self(), Sensitivity * Weight}}
        end,
        New_weights,
        convert_to_keys(Inputs)),

      perceptron(New_weights, Inputs, New_sensitivities);

    {connect_to_output, Receiver_PID} ->
      Combined_sensitivities = 
        [{Receiver_PID, 0} | Sensitivities],
      perceptron(Weights, Inputs, Combined_sensitivities);

    {connect_to_input, Sender_PID} ->
      Combined_input = [{Sender_PID, 0.5} | Inputs],
      perceptron([0.5 | Weights], Combined_input, Sensitivities);

    {pass, Input_value} ->
      lists:foreach(fun(Output_PID) ->
          Output_PID ! {stimulate, {self(), Input_value}}
        end,
        convert_to_keys(Sensitivities)),
      perceptron(Weights, Inputs, Sensitivities)

  end.

feed_forward(Func, Weights, Inputs) -> 
  Func(dot_prod(Weights, Inputs)).

add_sensitivity([], _Backprop) -> [];
add_sensitivity(Sensitivities, Backprop) -> 
  replace_input(Sensitivities, Backprop).

calculate_sensitivities(_Backprop, [], 
  _Sensitivities, _Output, _Deriv) ->
  null;
calculate_sensitivities({_, Training_value}, _Inputs, 
  [], Output, Deriv) ->
  (Training_value - Output) * Deriv;
calculate_sensitivities(_Backprop, _Inputs, 
  Sensitivities, _Output, Deriv) ->
  Deriv * lists:sum(convert_to_values(Sensitivities)).

connect(Sender, Receiver) ->
  ann_grapher ! {connect, Sender, Receiver},
  Sender ! {connect_to_output, Receiver},
  Receiver ! {connect_to_input, Sender}.

replace_input(Inputs, Input) ->
  {Input_PID, _} = Input,
  lists:keyreplace(Input_PID, 1, Inputs, Input).

convert_to_values(Tuple_list) ->
  lists:map(fun({_, Val}) -> Val end, Tuple_list).

convert_to_keys(Tuple_list) ->
  lists:map(fun({Key, _}) -> Key end, Tuple_list).

dot_prod(X, Y) -> dot_prod(0, X, Y).

dot_prod(Acc, [], []) -> Acc;
dot_prod(Acc, [X1|X], [Y1|Y]) ->
  dot_prod(X1*Y1 + Acc, X, Y).

vector_map(Func, X, Y) ->
  vector_map([], Func, X, Y).

vector_map(Acc, _Func, [], []) ->
  lists:reverse(Acc);
vector_map(Acc, Func, [Xh | Xt], [Yh | Yt]) ->
  vector_map([Func(Xh, Yh)|Acc], Func, Xt, Yt).
