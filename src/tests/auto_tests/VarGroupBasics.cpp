﻿#include <iostream>
#include "dynaplex/vargroup.h"
#include "dynaplex/error.h"
#include <gtest/gtest.h>



TEST(DynaPlexTests, VarGroupBasics) {



	DynaPlex::VarGroup vars({ {"p",2} , {"q",3.1} , { "s", "string"}, {"list", DynaPlex::VarGroup::DoubleVec{1.2,1.3}} });

	//this corresponds to the folowing json (note that VarGroup supports a specific subset of JSON):
	std::string json_string = R"(
	{
		"p": 2,
		"q": 3.1,
		"s": "string",
		"list": [1.2, 1.3]
	}
	)";
	//approach to construct from this string
	auto vars_from_string = DynaPlex::VarGroup(json_string);


	std::string valid_json_string = R"(
	{
		"list": [1.2, "asdf"]
	}
	)";

	//VarGroup does not support inhomogeneous lists
	EXPECT_THROW({
		DynaPlex::VarGroup test = DynaPlex::VarGroup(valid_json_string);
		}, DynaPlex::Error);

	//VarGroup does not support top-level lists (or simple objects).
	std::string list_string = R"(
	[0,1,2]
	)";
	EXPECT_THROW({
	DynaPlex::VarGroup test = DynaPlex::VarGroup(list_string);
		}, DynaPlex::Error);

	EXPECT_EQ(vars, vars_from_string);

	//building up by adding elements one by one:
	DynaPlex::VarGroup vars_alt = DynaPlex::VarGroup{};
	vars_alt.Add("p", 2);
	vars_alt.Add("q", 3.1);
	vars_alt.Add("s", "string");
	vars_alt.Add("list", DynaPlex::VarGroup::DoubleVec{ 1.2,1.3 });

	EXPECT_EQ(vars, vars_alt);

	


	DynaPlex::VarGroup different_order = DynaPlex::VarGroup{};
	different_order.Add("p", 2);
	different_order.Add("q", 3.1);
	different_order.Add("list", DynaPlex::VarGroup::DoubleVec{ 1.2,1.3 });
	different_order.Add("s", "string");
	//note that if vars are added in different order, they are no longer considered equal. 
	EXPECT_NE(vars_alt, different_order);
	

	vars.Add("listD", DynaPlex::VarGroup::DoubleVec{ 1.9,2.3 });

	int p;
	vars.Get("p", p);
	EXPECT_EQ(p, 2);

	double q;
	vars.Get("q", q);
	EXPECT_EQ(q, 3.1);

	std::string s;
	vars.Get("s", s);
	EXPECT_EQ(s, "string");

	std::vector<double> listD;
	std::vector<int> listI;
	vars.Get("listD", listD);
	EXPECT_EQ(listD[0], 1.9);

	//Note that doubles are silently truncated to ints when attempting to initiate an int with a double. 
	vars.Get("listD", listI);
	EXPECT_EQ(listI[0], 1);



	vars.Get("q", p);
	EXPECT_EQ(p, 3);

	EXPECT_THROW({
		vars.Get("p",s);
		}, DynaPlex::Error
	);


	EXPECT_THROW({
		vars.Get("not_available",s);
		}, DynaPlex::Error
	);

	auto vars2 = vars;

	std::string string = "test";
	vars.Add("new key", "value");

	EXPECT_NO_THROW(
		{
		vars.Get("new key", string);
		}
	);

	//vars2 is deep copy, so it should not contain the "new key" that was added after the copy was made. 
	EXPECT_THROW({
		vars2.Get("new key", s);
		}, DynaPlex::Error);



	auto vargroup1 = DynaPlex::VarGroup({ {"Id","tset"},{"Size",1.0} });

	auto vargroup2 = DynaPlex::VarGroup({ {"Id","tset"},{"Size",1.0} });

	auto list = ::DynaPlex::VarGroup::VarGroupVec{ vargroup1,vargroup2 };


	EXPECT_EQ(vargroup1, list[0]);
	EXPECT_EQ(vargroup2, list[1]);

	//This will not change the VarGroup in the list, as that was copied when the list was created.
	vargroup2.Add("as", 1);
	EXPECT_NE(vargroup2, list[1]);
}
