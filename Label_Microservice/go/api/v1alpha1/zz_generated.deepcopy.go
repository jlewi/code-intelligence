// +build !ignore_autogenerated

/*


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Code generated by controller-gen. DO NOT EDIT.

package v1alpha1

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// DeepCopyInto is an autogenerated deepcopy function, copying the receiver, writing into out. in must be non-nil.
func (in *ModelSync) DeepCopyInto(out *ModelSync) {
	*out = *in
	out.TypeMeta = in.TypeMeta
	in.ObjectMeta.DeepCopyInto(&out.ObjectMeta)
	in.Spec.DeepCopyInto(&out.Spec)
	in.Status.DeepCopyInto(&out.Status)
}

// DeepCopy is an autogenerated deepcopy function, copying the receiver, creating a new ModelSync.
func (in *ModelSync) DeepCopy() *ModelSync {
	if in == nil {
		return nil
	}
	out := new(ModelSync)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyObject is an autogenerated deepcopy function, copying the receiver, creating a new runtime.Object.
func (in *ModelSync) DeepCopyObject() runtime.Object {
	if c := in.DeepCopy(); c != nil {
		return c
	}
	return nil
}

// DeepCopyInto is an autogenerated deepcopy function, copying the receiver, writing into out. in must be non-nil.
func (in *ModelSyncList) DeepCopyInto(out *ModelSyncList) {
	*out = *in
	out.TypeMeta = in.TypeMeta
	in.ListMeta.DeepCopyInto(&out.ListMeta)
	if in.Items != nil {
		in, out := &in.Items, &out.Items
		*out = make([]ModelSync, len(*in))
		for i := range *in {
			(*in)[i].DeepCopyInto(&(*out)[i])
		}
	}
}

// DeepCopy is an autogenerated deepcopy function, copying the receiver, creating a new ModelSyncList.
func (in *ModelSyncList) DeepCopy() *ModelSyncList {
	if in == nil {
		return nil
	}
	out := new(ModelSyncList)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyObject is an autogenerated deepcopy function, copying the receiver, creating a new runtime.Object.
func (in *ModelSyncList) DeepCopyObject() runtime.Object {
	if c := in.DeepCopy(); c != nil {
		return c
	}
	return nil
}

// DeepCopyInto is an autogenerated deepcopy function, copying the receiver, writing into out. in must be non-nil.
func (in *ModelSyncSpec) DeepCopyInto(out *ModelSyncSpec) {
	*out = *in
	in.PipelineRunTemplate.DeepCopyInto(&out.PipelineRunTemplate)
	if in.SuccessfulPipelineRunsHistoryLimit != nil {
		in, out := &in.SuccessfulPipelineRunsHistoryLimit, &out.SuccessfulPipelineRunsHistoryLimit
		*out = new(int32)
		**out = **in
	}
	if in.FailedPipelineRunsHistoryLimit != nil {
		in, out := &in.FailedPipelineRunsHistoryLimit, &out.FailedPipelineRunsHistoryLimit
		*out = new(int32)
		**out = **in
	}
}

// DeepCopy is an autogenerated deepcopy function, copying the receiver, creating a new ModelSyncSpec.
func (in *ModelSyncSpec) DeepCopy() *ModelSyncSpec {
	if in == nil {
		return nil
	}
	out := new(ModelSyncSpec)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto is an autogenerated deepcopy function, copying the receiver, writing into out. in must be non-nil.
func (in *ModelSyncStatus) DeepCopyInto(out *ModelSyncStatus) {
	*out = *in
	if in.Active != nil {
		in, out := &in.Active, &out.Active
		*out = make([]v1.ObjectReference, len(*in))
		copy(*out, *in)
	}
}

// DeepCopy is an autogenerated deepcopy function, copying the receiver, creating a new ModelSyncStatus.
func (in *ModelSyncStatus) DeepCopy() *ModelSyncStatus {
	if in == nil {
		return nil
	}
	out := new(ModelSyncStatus)
	in.DeepCopyInto(out)
	return out
}

// DeepCopyInto is an autogenerated deepcopy function, copying the receiver, writing into out. in must be non-nil.
func (in *PipelineRunTemplate) DeepCopyInto(out *PipelineRunTemplate) {
	*out = *in
	in.ObjectMeta.DeepCopyInto(&out.ObjectMeta)
	in.Spec.DeepCopyInto(&out.Spec)
}

// DeepCopy is an autogenerated deepcopy function, copying the receiver, creating a new PipelineRunTemplate.
func (in *PipelineRunTemplate) DeepCopy() *PipelineRunTemplate {
	if in == nil {
		return nil
	}
	out := new(PipelineRunTemplate)
	in.DeepCopyInto(out)
	return out
}
