﻿#include <iostream>
#include "dynaplex/vargroup.h"
#include "dynaplex/error.h"
#include <gtest/gtest.h>
#include "SomeClass.h"


TEST(DynaPlexTests, InitiateClassWithVarGroup) {

	auto nested = DynaPlex::VarGroup({ {"Id","1"},{"Size",1.0} });

	auto nested2 = DynaPlex::VarGroup({ {"Id","2"},{"Size",4.0} });
	auto vectorofnestedelements = DynaPlex::VarGroup({ {"Id","2"},{"Size",2.0} });

	DynaPlex::VarGroup varGroup({
		{"myString","string"},
		{"myInt",42},
		{"myVector", DynaPlex::VarGroup::Int64Vec{123,123}},
		{ "nestedClass",nested},
		{"myNestedVector",DynaPlex::VarGroup::VarGroupVec{nested,nested2}}});


	//Initiate someclass with the varGroup
	SomeClass someclass(varGroup);

	//now someclass should have taken the values from the vargroup.
	ASSERT_EQ(someclass.myInt, 42);
	ASSERT_EQ(someclass.myString, "string");
	ASSERT_EQ(someclass.myVector.size(), 2);
	ASSERT_EQ(someclass.nestedClass.Size, 1.0);

	ASSERT_EQ(someclass.myNestedVector[1].Id, "2");




}