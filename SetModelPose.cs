using System.Collections;
using DilmerGames.Enums;
using DilmerGames.Core.Utilities;
using System.Collections.Generic;
using UnityEngine;


public class SetModelPose : MonoBehaviour
{
         private VNectModel.JointPoint[] thisjointPoints;
        private Animator anim;
    public List<Vector3> transform = new List<Vector3>(28);
    public List<Quaternion> rotation = new List<Quaternion>(28);

    public VNectModel VNectModel;
    private VNectModel.JointPoint[] jointPoints;
        public GameObject Nose;
    private bool yse;

    // Start is called before the first frame update
  
    void Start()
    {
        jointPoints = VNectModel.Init();
        if (jointPoints == null)
        {
            yse = true;
        }

        //    jointPoints = otherObject.GetComponent<VNectModel>().JointPoints;


        thisjointPoints = new VNectModel.JointPoint[PositionIndex.Count.Int()];
              for (var i = 0; i < PositionIndex.Count.Int(); i++) thisjointPoints[i] = new VNectModel.JointPoint();

        anim = this.gameObject.GetComponent<Animator>();
        thisjointPoints[PositionIndex.rShldrBend.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightUpperArm);
        thisjointPoints[PositionIndex.rForearmBend.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightLowerArm);
        thisjointPoints[PositionIndex.rHand.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightHand);
        thisjointPoints[PositionIndex.rThumb2.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightThumbIntermediate);
        thisjointPoints[PositionIndex.rMid1.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightMiddleProximal);
        // Left Arm
        thisjointPoints[PositionIndex.lShldrBend.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftUpperArm);
        thisjointPoints[PositionIndex.lForearmBend.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftLowerArm);
        thisjointPoints[PositionIndex.lHand.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftHand);
        thisjointPoints[PositionIndex.lThumb2.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftThumbIntermediate);
        thisjointPoints[PositionIndex.lMid1.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftMiddleProximal);

        // Face
        thisjointPoints[PositionIndex.lEar.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Head);
        thisjointPoints[PositionIndex.lEye.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftEye);
        thisjointPoints[PositionIndex.rEar.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Head);
        thisjointPoints[PositionIndex.rEye.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightEye);
        thisjointPoints[PositionIndex.Nose.Int()].Transform = Nose.transform;

        // Right Leg
        thisjointPoints[PositionIndex.rThighBend.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightUpperLeg);
        thisjointPoints[PositionIndex.rShin.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightLowerLeg);
        thisjointPoints[PositionIndex.rFoot.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightFoot);
        thisjointPoints[PositionIndex.rToe.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.RightToes);

        // Left Leg
        thisjointPoints[PositionIndex.lThighBend.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
        thisjointPoints[PositionIndex.lShin.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftLowerLeg);
        thisjointPoints[PositionIndex.lFoot.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftFoot);
        thisjointPoints[PositionIndex.lToe.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.LeftToes);

        // etc
        thisjointPoints[PositionIndex.abdomenUpper.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Spine);
        thisjointPoints[PositionIndex.hip.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Hips);
        thisjointPoints[PositionIndex.head.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Head);
        thisjointPoints[PositionIndex.neck.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Neck);
        thisjointPoints[PositionIndex.spine.Int()].Transform = anim.GetBoneTransform(HumanBodyBones.Spine);

    }

    // Update is called once per frame
    void LateUpdate()
    {

        TCPControllerClient.Instance.working = true;

        // Right Arm
        for (int i = 0 ; i<28;i++){

            if (i!=11&&i!=13){
                TCPControllerClient.Instance.AddModel(jointPoints[i].Transform.position);
                TCPControllerClient.Instance.AddModelRotation(jointPoints[i].Transform.rotation);
               // thisjointPoints[i].Transform.SetPositionAndRotation(jointPoints[i].Transform.position,jointPoints[i].Transform.rotation);

            }

        }
        TCPControllerClient.Instance.working = false;
    }





}
