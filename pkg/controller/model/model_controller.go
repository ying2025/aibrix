/*
Copyright 2024 The Aibrix Team.

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

package model

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	corelisters "k8s.io/client-go/listers/core/v1"
	toolscache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	modelv1alpha1 "github.com/aibrix/aibrix/api/model/v1alpha1"
)

const (
	ModelIdentifierKey                = "model.aibrix.ai/name"
	ModelAdapterFinalizer             = "adapter.model.aibrix.ai/finalizer"
	ModelAdapterPodTemplateLabelKey   = "adapter.model.aibrix.ai/enabled"
	ModelAdapterPodTemplateLabelValue = "true"

	// Reasons for model adapter conditions
	// Processing:

	// ModelAdapterInitializedReason is added in model adapter when it comes into the reconciliation loop.
	ModelAdapterInitializedReason = "ModelAdapterPending"
	// FailedServiceCreateReason is added in a model adapter when it cannot create a new service.
	FailedServiceCreateReason = "ServiceCreateError"
	// FailedEndpointSliceCreateReason is added in a model adapter when it cannot create a new replica set.
	FailedEndpointSliceCreateReason = "EndpointSliceCreateError"
	// ModelAdapterLoadingErrorReason is added in a model adapter when it cannot be loaded in an engine pod.
	ModelAdapterLoadingErrorReason = "ModelAdapterLoadingError"
	// ValidationFailedReason is added when model adapter object fails the validation
	ValidationFailedReason = "ValidationFailed"
	// StableInstanceFoundReason is added if there's stale pod and instance has been deleted successfully.
	StableInstanceFoundReason = "StableInstanceFound"

	// Available:

	// ModelAdapterAvailable is added in a ModelAdapter when it has replicas available.
	ModelAdapterAvailable = "ModelAdapterAvailable"
	// ModelAdapterUnavailable is added in a ModelAdapter when it doesn't have any pod hosting it.
	ModelAdapterUnavailable = "ModelAdapterUnavailable"
)

var (
	controllerKind         = modelv1alpha1.GroupVersion.WithKind("Model")
	controllerName         = "model-controller"
	defaultRequeueDuration = 3 * time.Second
)

// Add creates a new Model Controller and adds it to the Manager with default RBAC.
// The Manager will set fields on the Controller and Start it when the Manager is Started.
func Add(mgr manager.Manager) error {
	r, err := newReconciler(mgr)
	if err != nil {
		return err
	}
	return add(mgr, r)
}

// newReconciler returns a new reconcile.Reconciler
func newReconciler(mgr manager.Manager) (reconcile.Reconciler, error) {
	cacher := mgr.GetCache()

	podInformer, err := cacher.GetInformer(context.TODO(), &corev1.Pod{})
	if err != nil {
		return nil, err
	}

	serviceInformer, err := cacher.GetInformer(context.TODO(), &corev1.Service{})
	if err != nil {
		return nil, err
	}

	// Let's generate the clientset and use ModelAdapterLister here as well.
	podLister := corelisters.NewPodLister(podInformer.(toolscache.SharedIndexInformer).GetIndexer())
	serviceLister := corelisters.NewServiceLister(serviceInformer.(toolscache.SharedIndexInformer).GetIndexer())

	reconciler := &ModelReconciler{
		Client:        mgr.GetClient(),
		Scheme:        mgr.GetScheme(),
		PodLister:     podLister,
		ServiceLister: serviceLister,
		Recorder:      mgr.GetEventRecorderFor(controllerName),
	}
	return reconciler, nil
}

// add adds a new Controller to mgr with r as the reconcile.Reconciler
func add(mgr manager.Manager, r reconcile.Reconciler) error {
	// use the builder fashion. If we need more fine grain control later, we can switch to `controller.New()`
	err := ctrl.NewControllerManagedBy(mgr).
		Named(controllerName).
		For(&modelv1alpha1.Model{}, builder.WithPredicates(predicate.Or(
			predicate.GenerationChangedPredicate{},
			predicate.LabelChangedPredicate{},
			predicate.AnnotationChangedPredicate{},
		))).
		Complete(r)

	klog.V(4).InfoS("Finished to add model-adapter-controller")
	return err
}

var _ reconcile.Reconciler = &ModelReconciler{}

// ModelAdapterReconciler reconciles a ModelAdapter object
type ModelReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder

	// PodLister is able to list/get pods from a shared informer's cache store
	PodLister corelisters.PodLister
	// ServiceLister is able to list/get services from a shared informer's cache store
	ServiceLister corelisters.ServiceLister
	// EndpointSliceLister is able to list/get services from a shared informer's cache store
}

//+kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=services/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=apps,resources=deployments/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=core,resources=pods/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=model.aibrix.ai,resources=models,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=model.aibrix.ai,resources=models/status,verbs=get;update;patch

// Reconcile reads that state of ModelAdapter object and makes changes based on the state read
// and what is in the ModelAdapter.Spec
func (r *ModelReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	klog.V(4).InfoS("Starting to process Model", "model", req.NamespacedName)

	// Fetch the ModelAdapter instance
	model := &modelv1alpha1.Model{}
	err := r.Get(ctx, req.NamespacedName, model)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// Object not found, return.
			// For service, endpoint objects, clean up the resources using finalizers
			klog.InfoS("ModelAdapter resource not found. Ignoring since object mush be deleted", "modelAdapter", req.NamespacedName)
			return reconcile.Result{}, nil
		}

		// Error reading the object and let's requeue the request
		klog.ErrorS(err, "Failed to get Model", "Model", klog.KObj(model))
		return reconcile.Result{}, err
	}

	// if modelAdapter.ObjectMeta.DeletionTimestamp.IsZero() {
	// 	// the object is not being deleted, so if it does not have the finalizer,
	// 	// then lets add the finalizer and update the object.
	// 	if !controllerutil.ContainsFinalizer(modelAdapter, ModelAdapterFinalizer) {
	// 		klog.InfoS("Adding finalizer for ModelAdapter", "ModelAdapter", klog.KObj(modelAdapter))
	// 		if ok := controllerutil.AddFinalizer(modelAdapter, ModelAdapterFinalizer); !ok {
	// 			klog.Error("Failed to add finalizer for ModelAdapter")
	// 			return ctrl.Result{Requeue: true}, nil
	// 		}
	// 		if err := r.Update(ctx, modelAdapter); err != nil {
	// 			klog.Error("Failed to update custom resource to add finalizer")
	// 			return ctrl.Result{}, err
	// 		}
	// 	}
	// } else {
	// 	// the object is being deleted
	// 	if controllerutil.ContainsFinalizer(modelAdapter, ModelAdapterFinalizer) {
	// 		// the finalizer is present, so let's unload lora from those inference engines
	// 		// note: the base model pod could be deleted as well, so here we do best effort offloading
	// 		// we do not need to reconcile the object if it encounters the unloading error.
	// 		if err := r.unloadModelAdapter(modelAdapter); err != nil {
	// 			return ctrl.Result{}, err
	// 		}
	// 		if ok := controllerutil.RemoveFinalizer(modelAdapter, ModelAdapterFinalizer); !ok {
	// 			klog.Error("Failed to remove finalizer for ModelAdapter")
	// 			return ctrl.Result{Requeue: true}, nil
	// 		}
	// 		if err := r.Update(ctx, modelAdapter); err != nil {
	// 			klog.Error("Failed to update custom resource to remove finalizer")
	// 			return ctrl.Result{}, err
	// 		}
	// 	}
	// 	// Stop reconciliation as the item is being deleted
	// 	return ctrl.Result{}, nil
	// }

	return r.DoReconcile(ctx, req, model)
}

func (r *ModelReconciler) DoReconcile(ctx context.Context, req ctrl.Request, instance *modelv1alpha1.Model) (ctrl.Result, error) {
	// Let's set the initial status when no status is available
	if instance.Status.Conditions == nil || len(instance.Status.Conditions) == 0 {
		instance.Status.Phase = modelv1alpha1.ModelPending
		condition := NewCondition(string(modelv1alpha1.ModelConditionTypeInitialized), metav1.ConditionUnknown,
			ModelAdapterInitializedReason, "Starting reconciliation")
		if err := r.updateStatus(ctx, instance, condition); err != nil {
			return reconcile.Result{}, err
		} else {
			return reconcile.Result{Requeue: true}, nil
		}
	}

	oldInstance := instance.DeepCopy()
	// Step 0: Validate ModelAdapter configurations
	// if err := validateModelAdapter(instance); err != nil {
	// 	klog.Error(err, "Failed to validate the ModelAdapter")
	// 	instance.Status.Phase = modelv1alpha1.ModelAdapterFailed
	// 	condition := NewCondition(string(modelv1alpha1.ModelAdapterPending), metav1.ConditionFalse,
	// 		ValidationFailedReason, "ModelAdapter resource is not valid")
	// 	// TODO: no need to update the status if the status remain the same
	// 	if updateErr := r.updateStatus(ctx, instance, condition); updateErr != nil {
	// 		klog.ErrorS(err, "Failed to update ModelAdapter status", "modelAdapter", klog.KObj(instance))
	// 		return reconcile.Result{}, updateErr
	// 	}

	// 	return ctrl.Result{}, err
	// }

	// Step 3: Reconcile Service
	if ctrlResult, err := r.reconcileService(ctx, instance); err != nil {
		instance.Status.Phase = modelv1alpha1.ModelResourceCreated
		condition := NewCondition(string(modelv1alpha1.ModelConditionTypeResourceCreated), metav1.ConditionFalse,
			FailedServiceCreateReason, "service creation failure")
		if err := r.updateStatus(ctx, instance, condition); err != nil {
			klog.InfoS("Got error when updating status", req.Name, "error", err, "ModelAdapter", instance)
			return ctrl.Result{}, err
		}
		return ctrlResult, err
	}

	// Step 4: Reconcile EndpointSlice
	if ctrlResult, err := r.reconcileDeployment(ctx, instance); err != nil {
		instance.Status.Phase = modelv1alpha1.ModelResourceCreated
		condition := NewCondition(string(modelv1alpha1.ModelConditionTypeResourceCreated), metav1.ConditionFalse,
			FailedEndpointSliceCreateReason, "deployment creation failure")
		if err := r.updateStatus(ctx, instance, condition); err != nil {
			klog.InfoS("Got error when updating status", "error", err, "ModelAdapter", instance)
			return ctrl.Result{}, err
		}
		return ctrlResult, err
	}

	// Check if we need to update the status.
	if r.inconsistentModelStatus(oldInstance.Status, instance.Status) {
		condition := NewCondition(string(modelv1alpha1.ModelAdapterConditionReady), metav1.ConditionTrue,
			ModelAdapterAvailable, fmt.Sprintf("ModelAdapter %s is ready", klog.KObj(instance)))
		if err := r.updateStatus(ctx, instance, condition); err != nil {
			return reconcile.Result{}, fmt.Errorf("update modelAdapter status error: %v", err)
		}
	}

	return ctrl.Result{}, nil
}

func (r *ModelReconciler) reconcileService(ctx context.Context, instance *modelv1alpha1.Model) (ctrl.Result, error) {
	// Retrieve the Service from the Kubernetes cluster with the name and namespace.
	found := &corev1.Service{}
	err := r.Get(ctx, types.NamespacedName{Namespace: instance.Namespace, Name: instance.Name}, found)
	if err != nil && apierrors.IsNotFound(err) {
		// Service does not exist, create a new one
		svc := buildModelService(instance)
		// Set the owner reference
		if err := ctrl.SetControllerReference(instance, svc, r.Scheme); err != nil {
			klog.Error(err, "Failed to set controller reference to modelAdapter")
			return ctrl.Result{}, err
		}

		// create service
		klog.InfoS("Creating a new service", "service", klog.KObj(svc))
		if err = r.Create(ctx, svc); err != nil {
			klog.ErrorS(err, "Failed to create new service resource for ModelAdapter", "service", klog.KObj(svc))
			condition := NewCondition(string(modelv1alpha1.ModelAdapterConditionTypeResourceCreated), metav1.ConditionFalse,
				FailedServiceCreateReason, fmt.Sprintf("Failed to create Service for the modeladapter (%s): (%s)", klog.KObj(instance), err))
			if err := r.updateStatus(ctx, instance, condition); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{}, err
		}
	} else if err != nil {
		klog.ErrorS(err, "Failed to get Service")
		return ctrl.Result{}, err
	}
	// TODO: add `else` logic let's compare the service major fields and update to the target state.

	// TODO: Now, we are using the name comparison which is not enough,
	// compare the object difference in future.
	return ctrl.Result{}, nil
}

func buildModelService(instance *modelv1alpha1.Model) *corev1.Service {
	labels := map[string]string{
		"model.aibrix.ai/name": instance.Name,
	}

	ports := []corev1.ServicePort{
		{
			Name: "http",
			// it should use the base model service port.
			// make sure this can be dynamically configured later.
			Port: 8000,
			TargetPort: intstr.IntOrString{
				Type:   intstr.Int,
				IntVal: 8000,
			},
			Protocol: corev1.ProtocolTCP,
		},
	}

	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        instance.Name,
			Namespace:   instance.Namespace,
			Labels:      labels,
			Annotations: make(map[string]string),
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(instance, controllerKind),
			},
		},
		Spec: corev1.ServiceSpec{
			Selector:                 labels,
			ClusterIP:                corev1.ClusterIPNone,
			PublishNotReadyAddresses: true,
			Ports:                    ports,
		},
	}
}

func (r *ModelReconciler) reconcileDeployment(ctx context.Context, instance *modelv1alpha1.Model) (ctrl.Result, error) {
	// Retrieve the Service from the Kubernetes cluster with the name and namespace.
	found := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Namespace: instance.Namespace, Name: instance.Name}, found)
	if err != nil && apierrors.IsNotFound(err) {
		// Deployment does not exist, create a new one
		deployment := buildModelDeployment(instance)
		// Set the owner reference
		if err := ctrl.SetControllerReference(instance, deployment, r.Scheme); err != nil {
			klog.Error(err, "Failed to set controller reference to model")
			return ctrl.Result{}, err
		}

		// create deployment
		klog.InfoS("Creating a new deployment", "deployment", klog.KObj(deployment))
		if err = r.Create(ctx, deployment); err != nil {
			klog.ErrorS(err, "Failed to create new deployment resource for Model", "deployment", klog.KObj(deployment))
			condition := NewCondition(string(modelv1alpha1.ModelConditionTypeResourceCreated), metav1.ConditionFalse,
				FailedServiceCreateReason, fmt.Sprintf("Failed to create Deployment for the model (%s): (%s)", klog.KObj(instance), err))
			if err := r.updateStatus(ctx, instance, condition); err != nil {
				return ctrl.Result{}, err
			}
			return ctrl.Result{}, err
		}
	} else if err != nil {
		klog.ErrorS(err, "Failed to get deployment")
		return ctrl.Result{}, err
	}
	// TODO: add `else` logic let's compare the service major fields and update to the target state.

	// TODO: Now, we are using the name comparison which is not enough,
	// compare the object difference in future.
	return ctrl.Result{}, nil
}

func buildModelDeployment(instance *modelv1alpha1.Model) *appsv1.Deployment {
	podSelectorLabels := map[string]string{
		"model.aibrix.ai/name": instance.Name,
	}

	deploymentLabels := map[string]string{
		"model.aibrix.ai/name":            instance.Name,
		"model.aibrix.ai/port":            "8000",
		"adapter.model.aibrix.ai/enabled": "true",
	}

	podTemplate := instance.Spec.Template
	podTemplateLabels := podTemplate.Labels
	if len(podTemplateLabels) == 0 {
		podTemplateLabels = map[string]string{}
	}
	for k, v := range deploymentLabels {
		podTemplateLabels[k] = v
	}
	podTemplate.Labels = podTemplateLabels

	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:        instance.Name,
			Namespace:   instance.Namespace,
			Labels:      deploymentLabels,
			Annotations: make(map[string]string),
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(instance, controllerKind),
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: instance.Spec.Replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: podSelectorLabels,
			},
			Template: instance.Spec.Template,
			Strategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
			},
		},
	}
}

// NewCondition creates a new condition.
func NewCondition(condType string, status metav1.ConditionStatus, reason, msg string) metav1.Condition {
	return metav1.Condition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            msg,
	}
}

func (r *ModelReconciler) updateStatus(ctx context.Context, instance *modelv1alpha1.Model, condition metav1.Condition) error {
	klog.InfoS("model adapter reconcile", "Update CR status", instance.Name, "status", instance.Status)
	meta.SetStatusCondition(&instance.Status.Conditions, condition)
	return r.Status().Update(ctx, instance)
}

func (r *ModelReconciler) inconsistentModelStatus(oldStatus, newStatus modelv1alpha1.ModelStatus) bool {
	// Implement your logic to check if the status is inconsistent
	if oldStatus.Phase != newStatus.Phase {
		return true
	}

	return false
}
